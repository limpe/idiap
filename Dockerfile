FROM python:3.11-slim-bullseye

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y ffmpeg

COPY . .

CMD ["python", "main.py"]
