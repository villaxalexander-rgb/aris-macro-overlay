FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    unzip \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p logs data/signals docs/fund_notes

CMD ["python", "main.py"]