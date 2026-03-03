import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from crop_recommendation.entity.config_entity import ModelEvaluationConfig
from crop_recommendation.utils.logger import get_logger

logger = get_logger(__name__, "model_evaluation.log")


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def main_model_evaluation(self):
        logger.info("Starting model evaluation")

        model = joblib.load(self.config.model_path)

        test_df = pd.read_csv(self.config.processed_test_dir)
        X_test = test_df.drop(columns=["label"])
        y_test = test_df["label"]

        predictions = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average="weighted"),
            "recall": recall_score(y_test, predictions, average="weighted"),
            "f1_score": f1_score(y_test, predictions, average="weighted")
        }

        Path(self.config.metrics_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

        mlflow.set_tracking_uri("http://ec2-3-87-236-150.compute-1.amazonaws.com:5000")
        mlflow.set_experiment("Crop_Recommendation_Evaluation")

        with mlflow.start_run(run_name="Model_Evaluation"):
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(self.config.metrics_file)

        logger.info(f"Evaluation metrics: {metrics}")
        logger.info("Model evaluation completed")

        return metrics