import mlflow
import mlflow.pyfunc
import pandas as pd

mlflow.set_tracking_uri("http://ec2-54-91-16-63.compute-1.amazonaws.com:5000/")

model = mlflow.pyfunc.load_model("models:/random-forest@production")

sample = pd.DataFrame([{
    "N":90,
    "P":42,
    "K":43,
    "temperature":20.87974371,
    "humidity":82.00274423,
    "ph":6.502985292000001,
    "rainfall":202.9355362
}])

print(model.predict(sample))