FROM python:3.11-slim-bullseye

WORKDIR /app

COPY requirements.txt .  # Salin requirements.txt terlebih dahulu
RUN pip install -r requirements.txt # Instal dependencies

RUN apt-get update && apt-get install -y ffmpeg libavcodec-extra # Instal ffmpeg (penting untuk pydub)

COPY . . # Baru salin kode aplikasi Anda

CMD ["python", "main.py"]
