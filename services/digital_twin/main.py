"""
Digital Twin Service: EKF + CUSUM/IWD Authentication Pipeline.

Subscribes to the InfluxDB real-time sensor stream, runs the EKF state
estimator and the combined CUSUM/IWD authentication pipeline, and
publishes alarm events to a dedicated InfluxDB measurement.

From Sec. 4.5: "The Digital Twin container runs the EKF state estimator
and the full CUSUM/IWD pipeline in Python (NumPy/SciPy), subscribing to
the InfluxDB real-time stream and publishing alarm events to a dedicated
measurement, with end-to-end latency below 200 ms."
"""

import os
import sys
import time
import json
import logging
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

sys.path.insert(0, '/app')

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from core.config import SystemConfig, EKFConfig, CUSUMConfig, ISWTConfig
from core.ekf import ExtendedKalmanFilter
from core.cusum import CUSUMDetector
from core.iswt import ISWTDetector, combined_alarm
from core.calibration import calibrate_ekf, validate_whiteness

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'),
                    format='%(asctime)s [DT] %(levelname)s %(message)s')
log = logging.getLogger(__name__)

SENSOR_NAMES = ["L1", "L2", "P_in", "P_out", "Q12", "Q_pump"]


class DigitalTwinPipeline:
    """Real-time EKF + CUSUM/IWD pipeline."""

    def __init__(self):
        self.sys_config = SystemConfig()
        self.ekf_config = EKFConfig()
        self.cusum_config = CUSUMConfig(
            k=float(os.getenv('CUSUM_K', '0.5')),
            h=float(os.getenv('CUSUM_H', '5.0')),
        )
        self.iswt_config = ISWTConfig(
            W=int(os.getenv('ISWT_W', '200')),
            alpha=float(os.getenv('ISWT_ALPHA', '0.05')),
        )

        N = self.sys_config.n_sensors

        # Initialize components
        self.ekf = ExtendedKalmanFilter(self.sys_config, self.ekf_config)
        self.cusum = CUSUMDetector(N, self.cusum_config)
        self.iswt = ISWTDetector(N, self.iswt_config)

        # State
        self.step = 0
        self.calibrated = False
        self.u_prev = np.array([1.0, self.sys_config.valve_open])

        # Latest results (for status endpoint)
        self.latest = {
            'step': 0,
            'alarm': False,
            'cusum_G': np.zeros(N).tolist(),
            'cusum_alarm': [False] * N,
            'iswt_stat': 0.0,
            'iswt_alarm': False,
            'innovations': np.zeros(N).tolist(),
        }

    def process_measurement(self, y: np.ndarray,
                            u: np.ndarray) -> dict:
        """Run one step of the authentication pipeline.

        Args:
            y: (N,) sensor measurement vector.
            u: (2,) control input [u_pump, u_valve].

        Returns:
            Dictionary with alarm status and diagnostic info.
        """
        t_start = time.time()

        # EKF step
        ekf_result = self.ekf.step(y, u)

        # CUSUM update
        cusum_result = self.cusum.update(ekf_result['std_innovation'])

        # ISWT update
        iswt_result = self.iswt.update(ekf_result['std_innovation'])

        # Combined decision (Eq. 13)
        alarm = combined_alarm(cusum_result['alarm'], iswt_result['alarm'])

        latency_ms = (time.time() - t_start) * 1000

        result = {
            'step': self.step,
            'alarm': alarm,
            'cusum_G': cusum_result['G'].tolist(),
            'cusum_alarm': cusum_result['alarm'].tolist(),
            'iswt_test_stat': iswt_result['test_stat'],
            'iswt_critical': iswt_result['critical'],
            'iswt_alarm': iswt_result['alarm'],
            'iswt_ready': iswt_result['ready'],
            'innovations': ekf_result['innovation'].tolist(),
            'std_innovations': ekf_result['std_innovation'].tolist(),
            'latency_ms': latency_ms,
        }

        self.latest = result
        self.step += 1
        self.u_prev = u.copy()

        return result


class StatusHandler(BaseHTTPRequestHandler):
    """HTTP handler for DT status endpoint."""

    pipeline = None  # Set by main

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(self.pipeline.latest).encode())
        else:
            self.send_response(404)
            self.end_headers()


def main():
    log.info("=" * 60)
    log.info("Digital Twin: EKF + CUSUM/IWD Pipeline")
    log.info("=" * 60)

    # InfluxDB connection
    influx_url = os.getenv('INFLUXDB_URL', 'http://influxdb:8086')
    influx_token = os.getenv('INFLUXDB_TOKEN', 'cps-testbed-token')
    influx_org = os.getenv('INFLUXDB_ORG', 'cps-lab')
    sensor_bucket = os.getenv('INFLUXDB_BUCKET', 'sensors')
    alarm_bucket = os.getenv('ALARM_BUCKET', 'alarms')

    log.info(f"InfluxDB: {influx_url}")
    log.info(f"Sensor bucket: {sensor_bucket}, Alarm bucket: {alarm_bucket}")

    # Initialize pipeline
    pipeline = DigitalTwinPipeline()
    StatusHandler.pipeline = pipeline

    # Start status HTTP server
    status_port = int(os.getenv('STATUS_PORT', '9090'))
    status_server = HTTPServer(('0.0.0.0', status_port), StatusHandler)
    status_thread = threading.Thread(target=status_server.serve_forever,
                                      daemon=True)
    status_thread.start()
    log.info(f"Status endpoint: http://0.0.0.0:{status_port}/status")

    # Connect to InfluxDB
    influx_client = InfluxDBClient(
        url=influx_url, token=influx_token, org=influx_org
    )
    write_api = influx_client.write_api(write_options=SYNCHRONOUS)
    query_api = influx_client.query_api()

    # Wait for InfluxDB and initial data
    log.info("Waiting for InfluxDB and sensor data...")
    for _ in range(120):
        try:
            health = influx_client.health()
            if health.status == "pass":
                break
        except Exception:
            pass
        time.sleep(1)

    # Create alarm bucket if needed
    try:
        buckets_api = influx_client.buckets_api()
        existing = [b.name for b in buckets_api.find_buckets().buckets]
        if alarm_bucket not in existing:
            from influxdb_client import BucketRetentionRules
            buckets_api.create_bucket(
                bucket_name=alarm_bucket, org=influx_org
            )
            log.info(f"Created alarm bucket: {alarm_bucket}")
    except Exception as e:
        log.warning(f"Could not create alarm bucket: {e}")

    log.info("Starting real-time monitoring loop")

    last_step = -1
    while True:
        try:
            # Query latest sensor data
            query = f'''
            from(bucket: "{sensor_bucket}")
              |> range(start: -5s)
              |> filter(fn: (r) => r._measurement == "sensor_data")
              |> last()
            '''
            tables = query_api.query(query)

            # Parse fields into measurement vector
            fields = {}
            for table in tables:
                for record in table.records:
                    fields[record.get_field()] = record.get_value()

            if 'step' not in fields or fields.get('step', -1) == last_step:
                time.sleep(0.1)
                continue

            last_step = fields.get('step', -1)

            # Build measurement vector
            y = np.array([
                fields.get('L1', 2.5),
                fields.get('L2', 2.0),
                fields.get('P_in', 2.5),
                fields.get('P_out', 20.0),
                fields.get('Q12', 0.003),
                fields.get('Q_pump', 0.008),
            ])

            u = np.array([
                float(fields.get('u_pump', 1)),
                float(fields.get('u_valve', 60)) / 100.0,
            ])

            # Run pipeline
            result = pipeline.process_measurement(y, u)

            # Publish alarm events
            point = Point("alarm_data")
            point.field("alarm", int(result['alarm']))
            for i, name in enumerate(SENSOR_NAMES):
                point.field(f"G_{name}", result['cusum_G'][i])
                point.field(f"alarm_{name}", int(result['cusum_alarm'][i]))
                point.field(f"innov_{name}", result['innovations'][i])
            point.field("iswt_stat", result['iswt_test_stat'])
            point.field("iswt_alarm", int(result['iswt_alarm']))
            point.field("latency_ms", result['latency_ms'])
            point.field("step", result['step'])

            write_api.write(bucket=alarm_bucket, record=point)

            # Log alarms
            if result['alarm']:
                log.warning(f"ALARM at step {result['step']}! "
                            f"CUSUM={result['cusum_alarm']} "
                            f"ISWT={result['iswt_alarm']}")
            elif result['step'] % 60 == 0:
                log.info(f"Step {result['step']:5d} | OK | "
                         f"latency={result['latency_ms']:.1f}ms | "
                         f"ISWT={result['iswt_test_stat']:.2f}"
                         f"/{result['iswt_critical']:.2f}")

        except Exception as e:
            log.error(f"Pipeline error: {e}")

        time.sleep(0.1)


if __name__ == '__main__':
    main()
