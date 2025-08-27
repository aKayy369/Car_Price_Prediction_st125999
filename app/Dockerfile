# Simple Dockerfile for the Dash app
FROM python:3.11-slim

WORKDIR /app

# System deps (optional but good for pandas wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project (model should be created beforehand)
COPY . /app

# Expose port and run
EXPOSE 8050
CMD ["python", "app/app.py"]
