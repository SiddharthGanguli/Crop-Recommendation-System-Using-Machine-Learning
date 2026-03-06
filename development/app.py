from fastapi import FastAPI
import mlflow
import mlflow.pyfunc
import pandas as pd
import requests
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MLFLOW_TRACKING_URI = "http://44.205.205.0:5000/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = mlflow.pyfunc.load_model("models:/random-forest1@production")


LLM_URL = "http://13.216.161.236:11434/api/generate"

@app.get("/")
def health_check():
    return {"status": "API is running"}

def generate_explanation(prediction, data):

    prompt = f"""
You are an agricultural AI assistant.

Explain why the crop '{prediction}' is recommended based on these
conditions:

Nitrogen: {data['N']}
Phosphorus: {data['P']}
Potassium: {data['K']}
Temperature: {data['temperature']}
Humidity: {data['humidity']}
pH: {data['ph']}
Rainfall: {data['rainfall']}

Explain in simple terms for farmers in 3-4 short sentences.
"""

    response = requests.post(
        LLM_URL,
        json={
            "model": "phi3",
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()

    return result["response"]


@app.post("/predict")
def predict(data: dict):

    columns = [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]


    df = pd.DataFrame([data])[columns]


    prediction = model.predict(df)[0]

    explanation = generate_explanation(prediction, data)

    return {
        "prediction": prediction,
        "explanation": explanation
    }