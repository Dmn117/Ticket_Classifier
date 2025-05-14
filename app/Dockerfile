FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=22210

EXPOSE 22210

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
