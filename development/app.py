from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
import mlflow

app = FastAPI()

MLFLOW_TRACKING_URI = "http://ec2-3-87-236-150.compute-1.amazonaws.com:5000"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = mlflow.pyfunc.load_model("models:/random-forest@production")


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}