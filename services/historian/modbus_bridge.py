"""
Modbus-to-InfluxDB Bridge (Historian).

Polls all 6 holding registers from the process/proxy container at 1 Hz
via Modbus/TCP and writes timestamped values to InfluxDB.

From Sec. 4.5: "The historian container runs InfluxDB 2, recording all
Modbus register values at 1 Hz via a Python Modbus master bridge."
"""

import os
import sys
import time
import logging
from datetime import datetime

from pymodbus.client import ModbusTcpClient
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'),
                    format='%(asctime)s [HISTORIAN] %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# Sensor metadata for scaling
SENSOR_NAMES = ["L1", "L2", "P_in", "P_out", "Q12", "Q_pump"]
SENSOR_RANGES = [
    (0.0, 5.0),      # L1 [m]
    (0.0, 4.0),      # L2 [m]
    (0.0, 5.0),      # P_in [m_head]
    (0.0, 30.0),     # P_out [m_head]
    (-0.02, 0.02),   # Q12 [m³/s]
    (0.0, 0.02),     # Q_pump [m³/s]
]


def from_register(reg: int, low: float, high: float) -> float:
    """Scale a 16-bit Modbus register value to engineering units."""
    return low + (reg / 65535.0) * (high - low)


def main():
    log.info("=" * 60)
    log.info("Modbus-to-InfluxDB Bridge (Historian)")
    log.info("=" * 60)

    # Modbus connection
    mb_host = os.getenv('MODBUS_HOST', 'redteam-proxy')
    mb_port = int(os.getenv('MODBUS_PORT', '502'))

    # InfluxDB connection
    influx_url = os.getenv('INFLUXDB_URL', 'http://influxdb:8086')
    influx_token = os.getenv('INFLUXDB_TOKEN', 'cps-testbed-token')
    influx_org = os.getenv('INFLUXDB_ORG', 'cps-lab')
    influx_bucket = os.getenv('INFLUXDB_BUCKET', 'sensors')

    poll_interval = float(os.getenv('POLL_INTERVAL_S', '1.0'))

    log.info(f"Modbus: {mb_host}:{mb_port}")
    log.info(f"InfluxDB: {influx_url}, bucket={influx_bucket}")
    log.info(f"Poll interval: {poll_interval}s")

    # Initialize clients
    mb_client = ModbusTcpClient(mb_host, port=mb_port)
    influx_client = InfluxDBClient(
        url=influx_url, token=influx_token, org=influx_org
    )
    write_api = influx_client.write_api(write_options=SYNCHRONOUS)

    # Wait for InfluxDB to be ready
    log.info("Waiting for InfluxDB...")
    for _ in range(60):
        try:
            health = influx_client.health()
            if health.status == "pass":
                log.info("InfluxDB is ready")
                break
        except Exception:
            pass
        time.sleep(1)

    step = 0
    while True:
        try:
            if not mb_client.connected:
                mb_client.connect()
                time.sleep(0.5)
                continue

            # Read all sensor registers
            result = mb_client.read_holding_registers(
                address=0, count=6, slave=1
            )
            if result.isError():
                log.warning(f"Modbus read error: {result}")
                time.sleep(0.1)
                continue

            # Read actuator coils
            coil_result = mb_client.read_coils(address=0, count=2, slave=1)
            coils = coil_result.bits[:2] if not coil_result.isError() else [0, 0]

            # Scale to engineering units and write to InfluxDB
            point = Point("sensor_data")
            for i, name in enumerate(SENSOR_NAMES):
                lo, hi = SENSOR_RANGES[i]
                value = from_register(result.registers[i], lo, hi)
                point.field(name, value)
                # Also store raw register value
                point.field(f"{name}_raw", result.registers[i])

            # Add actuator states
            point.field("u_pump", int(coils[0]))
            point.field("u_valve", int(coils[1]))
            point.field("step", step)

            write_api.write(bucket=influx_bucket, record=point)

            if step % 60 == 0:
                values = [from_register(result.registers[i],
                                        SENSOR_RANGES[i][0],
                                        SENSOR_RANGES[i][1])
                          for i in range(6)]
                log.info(f"Step {step:5d} | "
                         + " ".join(f"{n}={v:.4f}"
                                   for n, v in zip(SENSOR_NAMES, values)))

            step += 1

        except Exception as e:
            log.error(f"Bridge error: {e}")
            try:
                mb_client.connect()
            except Exception:
                pass

        time.sleep(poll_interval)


if __name__ == '__main__':
    main()
