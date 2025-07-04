from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import requests
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputText(BaseModel):
    text: str

MODEL_URL = "https://drive.google.com/uc?export=download&id=1T_Ih3oDKiEMd8N1pZypzr0gxvxhezsz6"

try:
    response = requests.get(MODEL_URL)
    response.raise_for_status()
    model = joblib.load(BytesIO(response.content))
except Exception as e:
    model = None
    model_error = str(e)

@app.post("/BaggingML")
def predict_emotion(input: InputText):
    if model is None:
        return {"error": f"Model failed to load: {model_error}"}
    try:
        df = pd.DataFrame([input.text])
        pred = model.predict(df)
        return {"emotion": f"{pred[0]}"}
    except Exception as e:
        return {"error": str(e)}
