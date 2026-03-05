import joblib
import pandas as pd

model = joblib.load(r"/Users/siddharthaganguli/Desktop/Proj1/Ml_model/crop_recommendation_sys/artifacts/model_trainer/model.pkl")

df = pd.read_csv(r"/Users/siddharthaganguli/Desktop/Proj1/Ml_model/crop_recommendation_sys/artifacts/data_preprocessing/train_processed.csv")

X = df.drop(columns=["label"])

preds = model.predict(X)

print(pd.Series(preds).value_counts())


samples = pd.DataFrame([
# Rice
{"N":90,"P":40,"K":40,"temperature":26,"humidity":80,"ph":6.0,"rainfall":230},

# Banana
{"N":120,"P":40,"K":50,"temperature":30,"humidity":90,"ph":6.5,"rainfall":110},

# Coffee
{"N":100,"P":30,"K":30,"temperature":23,"humidity":60,"ph":6.5,"rainfall":150},

# Mango
{"N":20,"P":20,"K":20,"temperature":28,"humidity":70,"ph":6.5,"rainfall":80}
])

print(model.predict(samples))
print(model.predict_proba(samples))


print(model.feature_names_in_)

print(samples.columns)