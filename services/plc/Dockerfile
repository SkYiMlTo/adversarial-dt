FROM python:3.11-slim

# OpenPLC Runtime v3 is not trivially Dockerized for our use case.
# Instead, we implement the equivalent IEC 61131-3 control logic
# directly in Python, reading/writing Modbus registers exactly as
# OpenPLC would. This produces identical behavior on the Modbus
# bus while being lighter and more portable.
#
# The structured text control logic is provided in control_logic.st
# for reference and can be loaded into a real OpenPLC instance
# for validation.

WORKDIR /app
RUN pip install --no-cache-dir pymodbus>=3.5
COPY services/plc/main.py .
COPY services/plc/control_logic.st .
EXPOSE 502
CMD ["python", "-u", "main.py"]
