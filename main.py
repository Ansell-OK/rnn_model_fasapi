from fastapi import FastAPI
from pydantic import BaseModel
from model.model import predict_text


app = FastAPI()

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    Disease: str 

@app.get("/")
def home():
    return {'health_check': "Ok"}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    Disease = predict_text(payload.text)
    return {"Disease": Disease}