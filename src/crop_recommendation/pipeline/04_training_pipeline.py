import logging
from pathlib import Path
import mlflow

from crop_recommendation.configuration.config import ConfigManager
from crop_recommendation.components.training import ModelTrainer
from crop_recommendation.utils.logger import get_logger

STAGE_NAME = "Model Training Stage"

logger = get_logger(
    name=__name__,
    log_file="pipeline.log"
)


def main():
    logger.info(f"{STAGE_NAME} started.........")

    mlflow.set_tracking_uri("http://ec2-54-146-198-25.compute-1.amazonaws.com:5000")

    mlflow.set_experiment("Crop_Recommendation_Training")

    config_manager = ConfigManager(Path("config/config.yaml"))
    trainer_config = config_manager.get_model_trainer_config()

    trainer = ModelTrainer(config=trainer_config)
    trainer.main_model_trainer()

    logger.info(f"{STAGE_NAME} completed")


if __name__ == "__main__":
    main()