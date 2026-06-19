FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy>=1.24 scipy>=1.10 \
    influxdb-client>=1.36
COPY core/ /app/core/
COPY services/digital_twin/main.py .
EXPOSE 9090
CMD ["python", "-u", "main.py"]
