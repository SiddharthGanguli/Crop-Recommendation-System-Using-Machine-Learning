import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from crop_recommendation.entity.config_entity import DataIngestionConfig
from crop_recommendation.utils.logger import get_logger

logger = get_logger(
    name=__name__,
    log_file="data_ingestion.log"
)


class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _read_data(self) -> pd.DataFrame:
        if not self.config.source_dir.exists():
            raise FileNotFoundError(f"Source file not found at {self.config.source_dir}")
        
        logger.info("Reading dataset...")
        df = pd.read_csv(self.config.source_dir)
        logger.info(f"Dataset loaded with shape: {df.shape}")
        return df

    def _save_raw_data(self, df: pd.DataFrame) -> Path:
        raw_data_path = Path(self.config.root_dir) / "raw.csv"
        raw_data_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(raw_data_path, index=False)
        logger.info(f"Raw data saved at {raw_data_path}")

        return raw_data_path

    def _split_data(self, df: pd.DataFrame):
        logger.info("Splitting dataset into train and test...")
        return train_test_split(df, test_size=0.2, random_state=42)

    def run(self):
        try:
            logger.info("Data ingestion started.")

            os.makedirs(self.config.root_dir, exist_ok=True)

            df = self._read_data()
            self._save_raw_data(df)

            train_df, test_df = self._split_data(df)

            # Ensure directories exist
            self.config.train_dir.parent.mkdir(parents=True, exist_ok=True)
            self.config.test_dir.parent.mkdir(parents=True, exist_ok=True)

            train_df.to_csv(self.config.train_dir, index=False)
            test_df.to_csv(self.config.test_dir, index=False)

            logger.info("Train and test data saved successfully.")
            logger.info("Data ingestion completed.")

            return self.config.train_dir, self.config.test_dir

        except Exception as e:
            logger.exception("Error occurred during data ingestion.")
            raise e