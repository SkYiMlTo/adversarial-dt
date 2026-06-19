"""
Process Simulator: Two-tank ODE + Modbus/TCP Server.

Runs the two-tank water distribution dynamics as a closed-loop ODE
simulation, exposing state through Modbus/TCP holding registers at 1 Hz.

From the perspective of every other component (PLC, historian, DT,
attacker), the interaction is identical to that with a physical plant.

Modbus Register Map:
    Holding Registers (read-only for clients):
        0: L1      — Main tank level [scaled 16-bit]
        1: L2      — Secondary tank level [scaled 16-bit]
        2: P_in    — Pump inlet pressure [scaled 16-bit]
        3: P_out   — Pump outlet pressure [scaled 16-bit]
        4: Q12     — Inter-tank flow rate [scaled 16-bit]
        5: Q_pump  — Pump discharge flow rate [scaled 16-bit]

    Coil Registers (writable by PLC):
        0: u_pump  — Pump enable (0/1)
        1: u_valve — Valve position (0/100 mapped to 0.0-1.0)
"""

import os
import sys
import time
import struct
import logging
import threading
import numpy as np

# Add parent for core imports
sys.path.insert(0, '/app')

from pymodbus.server import StartTcpServer
from pymodbus.datastore import (
    ModbusSequentialDataBlock,
    ModbusSlaveContext,
    ModbusServerContext,
)

from core.config import SystemConfig
from core.process_model import TwoTankProcess

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'),
                    format='%(asctime)s [PROCESS] %(levelname)s %(message)s')
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engineering-unit ↔ Modbus 16-bit scaling
# ---------------------------------------------------------------------------

def to_register(value: float, low: float, high: float) -> int:
    """Scale a float to a 16-bit Modbus register value."""
    normalized = (value - low) / (high - low + 1e-12)
    normalized = max(0.0, min(1.0, normalized))
    return int(normalized * 65535)


def from_register(reg: int, low: float, high: float) -> float:
    """Scale a 16-bit Modbus register value back to engineering units."""
    return low + (reg / 65535.0) * (high - low)


class ProcessSimulator:
    """Runs the ODE in a background thread and updates Modbus registers."""

    def __init__(self, config: SystemConfig, context: ModbusSlaveContext):
        self.cfg = config
        self.process = TwoTankProcess(config)
        self.context = context
        self.x = config.x0.copy()
        self.u = np.array([1.0, config.valve_open])
        self.running = True

        # Seed for reproducibility
        seed = int(os.getenv('RANDOM_SEED', '42'))
        self.process.set_seed(seed)

    def run(self):
        """Main simulation loop: integrate ODE + update registers at 1 Hz."""
        log.info("Starting process simulation loop")
        log.info(f"Initial state: {self.x}")
        log.info(f"Sample period: {self.cfg.dt_sample}s, "
                 f"ODE step: {self.cfg.dt_sim}s")

        step = 0
        while self.running:
            t_start = time.time()

            # Read actuator commands from coil registers
            coils = self.context[0x01].getValues(1, count=2)
            self.u[0] = float(coils[0])           # u_pump: 0 or 1
            self.u[1] = float(coils[1]) / 100.0   # u_valve: 0-100 → 0.0-1.0

            # Integrate ODE for one sample period
            self.x = self.process.integrate(self.x, self.u)

            # Add measurement noise
            y = self.process.measure(self.x, add_noise=True)

            # Write sensor readings to holding registers
            ranges = self.cfg.sensor_ranges
            for i in range(self.cfg.n_sensors):
                reg_val = to_register(y[i], ranges[i][0], ranges[i][1])
                self.context[0x03].setValues(i + 1, [reg_val])

            if step % 60 == 0:
                log.info(f"Step {step:5d} | "
                         f"L1={y[0]:.3f}m L2={y[1]:.3f}m "
                         f"Pin={y[2]:.3f} Pout={y[3]:.2f} "
                         f"Q12={y[4]:.5f} Qp={y[5]:.5f} | "
                         f"u_pump={self.u[0]:.0f} u_valve={self.u[1]:.2f}")

            step += 1

            # Sleep to maintain 1 Hz sample rate
            elapsed = time.time() - t_start
            sleep_time = max(0, self.cfg.dt_sample - elapsed)
            time.sleep(sleep_time)


def main():
    log.info("=" * 60)
    log.info("Two-Tank Water Distribution Process Simulator")
    log.info("=" * 60)

    config = SystemConfig()

    # Initialize Modbus data store
    # Holding registers: 6 sensor values (addresses 1-6)
    hr_block = ModbusSequentialDataBlock(1, [0] * 10)

    # Coil registers: 2 actuator commands (addresses 1-2)
    # Initialize: pump ON (1), valve at 60% (60)
    coil_block = ModbusSequentialDataBlock(1, [1, 60] + [0] * 8)

    # Discrete inputs / input registers (unused, but required)
    di_block = ModbusSequentialDataBlock(1, [0] * 10)
    ir_block = ModbusSequentialDataBlock(1, [0] * 10)

    slave_context = ModbusSlaveContext(
        di=di_block, co=coil_block, hr=hr_block, ir=ir_block
    )
    server_context = ModbusServerContext(slaves=slave_context, single=True)

    # Start simulation in background thread
    simulator = ProcessSimulator(config, slave_context)
    sim_thread = threading.Thread(target=simulator.run, daemon=True)
    sim_thread.start()

    # Start Modbus TCP server (blocking)
    port = int(os.getenv('MODBUS_PORT', '502'))
    log.info(f"Starting Modbus/TCP server on port {port}")
    StartTcpServer(context=server_context, address=("0.0.0.0", port))


if __name__ == '__main__':
    main()
