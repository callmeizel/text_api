from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputText(BaseModel):
    text: str

model = joblib.load("model.joblib")

@app.post("/BaggingML")
def predict_emotion(input: InputText):
    df = pd.DataFrame([input.text])
    pred = model.predict(df)
    return {"emotion": f"{pred[0]}"}
