from pathlib import Path

from crop_recommendation.components.data_ingestion import DataIngestion
from crop_recommendation.configuration.config import ConfigManager
from crop_recommendation.utils.logger import get_logger

logger = get_logger(
    name=__name__,
    log_file="pipeline.log"
)

STAGE_NAME = "Data Ingestion Stage"


def main():
    try:
        logger.info(f"{STAGE_NAME} started...")

        config_manager = ConfigManager(
            file_path=Path("config/config.yaml")
        )

        data_ingestion_config = config_manager.get_data_ingestion_config()

        data_ingestion = DataIngestion(config=data_ingestion_config)

        train_path, test_path = data_ingestion.run()

        logger.info(f"Train data saved at: {train_path}")
        logger.info(f"Test data saved at: {test_path}")

        logger.info(f"{STAGE_NAME} completed successfully.")

    except Exception as e:
        logger.exception(f"Error in {STAGE_NAME}")
        raise e


if __name__ == "__main__":
    main()