FROM python:3.11-slim
WORKDIR /app
COPY services/process/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY core/ /app/core/
COPY services/process/main.py .
EXPOSE 502
CMD ["python", "-u", "main.py"]
