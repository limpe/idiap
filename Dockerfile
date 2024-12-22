FROM python:3.11-slim-bullseye

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libavcodec-extra \
    && rm -rf /var/lib/apt/lists/*

COPY . .

CMD ["python", "main.py"]
