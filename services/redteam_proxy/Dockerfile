FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir pymodbus>=3.5
COPY core/ /app/core/
COPY services/redteam_proxy/main.py .
EXPOSE 502 8080
CMD ["python", "-u", "main.py"]
