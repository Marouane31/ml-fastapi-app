from fastapi import FastAPI 
from pydantic import BaseModel
from app.predict import predict_iris

app = FastAPI()

class IrisFeatures(BaseModel):
    features:list

@app.post("/predict")
def get_prediction(data: IrisFeatures):
    prediction = predict_iris(data.features)
    return {"prediction": prediction}