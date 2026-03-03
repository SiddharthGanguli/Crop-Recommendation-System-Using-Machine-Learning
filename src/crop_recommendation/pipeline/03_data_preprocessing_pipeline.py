from pathlib import Path

from crop_recommendation.configuration.config import ConfigManager
from crop_recommendation.components.data_preprocessing import DataPreprocessing
from crop_recommendation.utils.logger import get_logger

logger = get_logger(
    name=__name__,
    log_file="pipeline.log"
)

STAGE_NAME = "Data Preprocessing Stage"


def main():
    try:
        logger.info(f"{STAGE_NAME} started...")

        config_manager = ConfigManager(
            file_path=Path("config/config.yaml")
        )

        preprocessing_config = config_manager.get_data_preprocessing_config()

        preprocessing = DataPreprocessing(config=preprocessing_config)

        train_path, test_path, scaler_path = preprocessing.run()

        logger.info(f"Processed train saved at: {train_path}")
        logger.info(f"Processed test saved at: {test_path}")
        logger.info(f"Scaler saved at: {scaler_path}")

        logger.info(f"{STAGE_NAME} completed successfully.")

    except Exception as e:
        logger.exception(f"Error in {STAGE_NAME}")
        raise e


if __name__ == "__main__":
    main() 