FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir pymodbus>=3.5 influxdb-client>=1.36
COPY services/historian/modbus_bridge.py .
CMD ["python", "-u", "modbus_bridge.py"]
