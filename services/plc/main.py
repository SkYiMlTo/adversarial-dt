"""
PLC Controller: Modbus/TCP client implementing IEC 61131-3 control logic.

Functionally equivalent to the Structured Text program in control_logic.st,
executed by polling Modbus/TCP registers at 1 Hz and writing actuator commands.

From Sec. 4.5: "OpenPLC executes real compiled IEC 61131-3 structured-text
control logic... polling the process container's Modbus/TCP registers at
1 Hz and writing actuator commands back, creating a genuine closed-loop
ICS control stack over a live TCP/IP network."
"""

import os
import sys
import time
import logging

from pymodbus.client import ModbusTcpClient

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'),
                    format='%(asctime)s [PLC] %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# Control thresholds
L1_LOW = 1.0      # Pump ON threshold [m]
L1_HIGH = 4.0     # Pump OFF threshold [m]
L1_SCALE = 5.0    # Full scale for L1
VALVE_POS = 60    # Valve position (0-100)


def main():
    log.info("=" * 60)
    log.info("PLC Controller (IEC 61131-3 equivalent)")
    log.info("=" * 60)

    host = os.getenv('MODBUS_HOST', 'redteam-proxy')
    port = int(os.getenv('MODBUS_PORT', '502'))
    cycle_ms = int(os.getenv('CYCLE_TIME_MS', '1000'))

    log.info(f"Connecting to Modbus server at {host}:{port}")
    log.info(f"Cycle time: {cycle_ms}ms")

    client = ModbusTcpClient(host, port=port)
    pump_cmd = True  # Initial state: pump ON

    while True:
        try:
            if not client.connected:
                client.connect()
                log.info("Connected to Modbus server")
                time.sleep(0.5)
                continue

            # Read sensor registers
            result = client.read_holding_registers(address=0, count=6, slave=1)
            if result.isError():
                log.warning(f"Read error: {result}")
                time.sleep(0.1)
                continue

            regs = result.registers

            # Scale L1 to engineering units
            L1 = (regs[0] / 65535.0) * L1_SCALE

            # Hysteresis pump control
            if L1 < L1_LOW:
                pump_cmd = True
            elif L1 > L1_HIGH:
                pump_cmd = False

            # Write actuator commands
            client.write_coils(0, [pump_cmd, VALVE_POS], slave=1)

        except Exception as e:
            log.error(f"PLC cycle error: {e}")
            try:
                client.connect()
            except Exception:
                pass

        time.sleep(cycle_ms / 1000.0)


if __name__ == '__main__':
    main()
