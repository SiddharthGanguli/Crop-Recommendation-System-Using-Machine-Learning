from fastapi import FastAPI
import mlflow
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

MLFLOW_TRACKING_URI = "http://44.205.205.0:5000/"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = mlflow.pyfunc.load_model("models:/random-forest1@production")


@app.get("/")
def health_check():
    return {"status": "API is running"}


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

    prediction = model.predict(df)

    return {"prediction": prediction.tolist()}