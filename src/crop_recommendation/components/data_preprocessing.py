import pandas as pd
from pathlib import Path

from crop_recommendation.entity.config_entity import DataPreprocessingConfig
from crop_recommendation.utils.logger import get_logger

logger = get_logger(
    name=__name__,
    log_file="data_preprocessing.log"
)


class DataPreprocessing:

    def __init__(self, config: DataPreprocessingConfig):
        self.config = config


    def _load_data(self):

        train_df = pd.read_csv(self.config.train_dir)
        test_df = pd.read_csv(self.config.test_dir)

        logger.info(f"Train shape: {train_df.shape}")
        logger.info(f"Test shape: {test_df.shape}")

        return train_df, test_df


    def run(self):

        logger.info("Data preprocessing started")

        train_df, test_df = self._load_data()

        Path(self.config.processed_train_dir).parent.mkdir(
            parents=True, exist_ok=True
        )

        train_df.to_csv(self.config.processed_train_dir, index=False)
        test_df.to_csv(self.config.processed_test_dir, index=False)

        logger.info("Data preprocessing completed")

        return (
            self.config.processed_train_dir,
            self.config.processed_test_dir
        )