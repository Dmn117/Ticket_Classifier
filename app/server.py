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


model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)


training_data = []


class ClassificationData(BaseModel):
    description: str

class TrainingData(BaseModel):
    description: str
    label: int



@app.post("/classify")
def classify(entry: ClassificationData, current_user: dict = Depends(get_current_user)):
    result = classifier(entry.description)
    return {"classification": result[0]["label"], "score": result[0]["score"]}



@app.post("/train")
def train(entry: List[TrainingData], current_user: dict = Depends(get_current_user)):
    global model, classifier, training_data

    if len(entry) < 2:
        raise HTTPException(status_code=400, detail="Se requieren al menos 2 ejemplos para entrenar.")

    training_data = entry

    dataset = Dataset.from_dict({
        "text": [d.description for d in training_data],
        "label": [d.label for d in training_data]
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

    num_labels=max(d.label for d in training_data) + 1

    new_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        id2label={i: str(i) for i in range(num_labels)},
        label2id={str(i): i for i in range(num_labels)}
    )

    trainer = Trainer(
        model=new_model,
        args=args,
        train_dataset=tokenized
    )

    trainer.train()

    model = new_model
    classifier = pipeline("text-classification", model=new_model, tokenizer=tokenizer, return_all_scores=False)

    return {"message": "Modelo reentrenado con éxito", "examples": len(training_data)}