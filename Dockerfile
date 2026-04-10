FROM python:3.10-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir uvicorn fastapi supervisor

COPY . .

RUN mkdir -p /app/artifacts && chmod -R 777 /app/artifacts

EXPOSE 7860

CMD ["supervisord", "-c", "supervisord.conf"]
