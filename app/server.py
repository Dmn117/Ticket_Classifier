import torch
from typing import List
from datasets import Dataset
from auth import verify_token
from pydantic import BaseModel
from fastapi.openapi.utils import get_openapi
from fastapi.security import OAuth2PasswordBearer
from fastapi import FastAPI, HTTPException, Depends
from transformers import pipeline, TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_current_user(token: str = Depends(oauth2_scheme)):
    return verify_token(token)

app = FastAPI()


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Ticket Classifier API",
        version="1.0.0",
        description="API que clasifica y reentrena modelos sobre descripciones de tickets.",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    openapi_schema["security"] = [{"bearerAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi


modelo_nombre = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(modelo_nombre)
modelo = AutoModelForSequenceClassification.from_pretrained(modelo_nombre, num_labels=2)
clasificador = pipeline("text-classification", model=modelo, tokenizer=tokenizer)


datos_entrenamiento = []


class EntradaClasificacion(BaseModel):
    descripcion: str

class EntradaEntrenamiento(BaseModel):
    descripcion: str
    etiqueta: int



@app.post("/clasificar")
def clasificar_ticket(entrada: EntradaClasificacion, current_user: dict = Depends(get_current_user)):
    resultado = clasificador(entrada.descripcion)
    return {"clasificacion": resultado[0]["label"], "score": resultado[0]["score"]}



@app.post("/entrenar")
def entrenar_nuevo_modelo(datos: List[EntradaEntrenamiento], current_user: dict = Depends(get_current_user)):
    global modelo, clasificador, datos_entrenamiento

    if len(datos) < 2:
        raise HTTPException(status_code=400, detail="Se requieren al menos 2 ejemplos para entrenar.")

    datos_entrenamiento = datos

    dataset = Dataset.from_dict({
        "text": [d.descripcion for d in datos_entrenamiento],
        "label": [d.etiqueta for d in datos_entrenamiento]
    })

    def tokenize_fn(example):
        return tokenizer(example["text"], truncation=True, padding=True)

    tokenized = dataset.map(tokenize_fn, batched=True)

    args = TrainingArguments(
        output_dir="./tmp_modelo",
        per_device_train_batch_size=4,
        num_train_epochs=2,
        logging_steps=10,
        save_strategy="no"
    )

    num_labels=max(d.etiqueta for d in datos_entrenamiento) + 1

    nuevo_modelo = AutoModelForSequenceClassification.from_pretrained(
        modelo_nombre, 
        num_labels=num_labels,
        id2label={i: str(i) for i in range(num_labels)},
        label2id={str(i): i for i in range(num_labels)}
    )

    trainer = Trainer(
        model=nuevo_modelo,
        args=args,
        train_dataset=tokenized
    )

    trainer.train()

    modelo = nuevo_modelo
    clasificador = pipeline("text-classification", model=nuevo_modelo, tokenizer=tokenizer, return_all_scores=False)

    return {"mensaje": "Modelo reentrenado con éxito", "ejemplos": len(datos_entrenamiento)}
