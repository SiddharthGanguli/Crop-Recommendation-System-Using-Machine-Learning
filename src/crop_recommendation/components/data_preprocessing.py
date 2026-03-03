import pandas as pd
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler

from crop_recommendation.entity.config_entity import DataPreprocessingConfig
from crop_recommendation.utils.logger import get_logger

logger = get_logger(
    name=__name__,
    log_file="data_preprocessing.log"
)


class DataPreprocessing:

    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
        self.scaler = StandardScaler()

    def _load_data(self):
        if not self.config.train_dir.exists():
            raise FileNotFoundError(f"Train file not found at {self.config.train_dir}")

        if not self.config.test_dir.exists():
            raise FileNotFoundError(f"Test file not found at {self.config.test_dir}")

        train_df = pd.read_csv(self.config.train_dir)
        test_df = pd.read_csv(self.config.test_dir)

        logger.info(f"Train shape: {train_df.shape}")
        logger.info(f"Test shape: {test_df.shape}")

        return train_df, test_df

    def _split_features_target(self, df):
        if "label" not in df.columns:
            raise ValueError("Target column 'label' not found in dataset")

        X = df.drop(columns=["label"])
        y = df["label"]
        return X, y

    def run(self):
        try:
            logger.info("Data preprocessing started.")

            train_df, test_df = self._load_data()

            X_train, y_train = self._split_features_target(train_df)
            X_test, y_test = self._split_features_target(test_df)

            # Fit only on train
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            train_processed = pd.DataFrame(
                X_train_scaled, columns=X_train.columns
            )
            train_processed["label"] = y_train.values

            test_processed = pd.DataFrame(
                X_test_scaled, columns=X_test.columns
            )
            test_processed["label"] = y_test.values

            # Ensure directories exist
            self.config.processed_train_dir.parent.mkdir(parents=True, exist_ok=True)
            self.config.scaler_path.parent.mkdir(parents=True, exist_ok=True)

            train_processed.to_csv(self.config.processed_train_dir, index=False)
            test_processed.to_csv(self.config.processed_test_dir, index=False)

            joblib.dump(self.scaler, self.config.scaler_path)

            logger.info("Data preprocessing completed successfully.")

            return (
                self.config.processed_train_dir,
                self.config.processed_test_dir,
                self.config.scaler_path,
            )

        except Exception as e:
            logger.exception("Error during data preprocessing.")
            raise e 