import pandas as pd
import yaml
import joblib
import optuna
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from crop_recommendation.entity.config_entity import ModelTrainerConfig
from crop_recommendation.utils.logger import get_logger


logger = get_logger(name=__name__, log_file="model_trainer.log")


class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    # -----------------------------
    # Load hyperparameters
    # -----------------------------
    def _load_params(self):

        with open(self.config.params_file, "r") as f:
            params = yaml.safe_load(f)

        return params


    # -----------------------------
    # Load dataset
    # -----------------------------
    def _load_data(self):

        train_df = pd.read_csv(self.config.processed_train_dir)
        test_df = pd.read_csv(self.config.processed_test_dir)

        logger.info(f"Train dataset size: {train_df.shape}")
        logger.info(f"Test dataset size: {test_df.shape}")
        logger.info(f"Classes: {train_df['label'].unique()}")

        return train_df, test_df


    # -----------------------------
    # Split features / target
    # -----------------------------
    def _split_features_target(self, df):

        X = df.drop(columns=["label"])
        y = df["label"]

        return X, y


    # -----------------------------
    # Optuna objective function
    # -----------------------------
    def _objective(self, trial, X_train, y_train, X_test, y_test, rf_params):

        params = {

            "n_estimators": trial.suggest_int(
                "n_estimators",
                rf_params["n_estimators"]["low"],
                rf_params["n_estimators"]["high"]
            ),

            "max_depth": trial.suggest_int(
                "max_depth",
                rf_params["max_depth"]["low"],
                rf_params["max_depth"]["high"]
            ),

            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                rf_params["min_samples_split"]["low"],
                rf_params["min_samples_split"]["high"]
            ),

            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                rf_params["min_samples_leaf"]["low"],
                rf_params["min_samples_leaf"]["high"]
            ),

            "random_state": rf_params["random_state"]
        }

        with mlflow.start_run(nested=True):

            model = RandomForestClassifier(
                **params,
                class_weight="balanced"
            )

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            accuracy = accuracy_score(y_test, preds)

            mlflow.log_params(params)
            mlflow.log_metric("accuracy", accuracy)

        return accuracy


    # -----------------------------
    # Main training pipeline
    # -----------------------------
    def main_model_trainer(self):

        logger.info("Starting model training")

        mlflow.set_tracking_uri(
            "http://ec2-54-91-16-63.compute-1.amazonaws.com:5000"
        )

        mlflow.set_experiment("Crop_Recommendation_Optuna")

        params = self._load_params()

        rf_params = params["random_forest"]
        n_trials = params["optuna"]["n_trials"]

        train_df, test_df = self._load_data()

        X_train, y_train = self._split_features_target(train_df)
        X_test, y_test = self._split_features_target(test_df)

        with mlflow.start_run(run_name="optuna_parent_run"):

            study = optuna.create_study(direction="maximize")

            study.optimize(
                lambda trial: self._objective(
                    trial,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    rf_params
                ),
                n_trials=n_trials
            )

            best_params = study.best_params
            best_params["random_state"] = rf_params["random_state"]

            logger.info(f"Best parameters: {best_params}")

            # -----------------------------
            # Train final model
            # -----------------------------
            model = RandomForestClassifier(
                **best_params,
                class_weight="balanced"
            )

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            final_accuracy = accuracy_score(y_test, preds)

            logger.info(f"Final model accuracy: {final_accuracy}")

            mlflow.log_params(best_params)
            mlflow.log_metric("final_accuracy", final_accuracy)

            # Log model to MLflow
            mlflow.sklearn.log_model(
                model,
                artifact_path="randomforestmodel"
            )

            # Save locally
            Path(self.config.model_path).parent.mkdir(
                parents=True,
                exist_ok=True
            )

            joblib.dump(model, self.config.model_path)

        logger.info("Model training completed")

        return self.config.model_path