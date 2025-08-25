FROM python:3.11-slim

WORKDIR /app

COPY ./app ./

COPY ./keys/public.pem ./

COPY ./LICENSE ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 22203

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 22203"]
