from pathlib import Path

from crop_recommendation.configuration.config import ConfigManager
from crop_recommendation.components.evaluation import ModelEvaluation
from crop_recommendation.utils.logger import get_logger


STAGE_NAME = "Model Evaluation Stage"

logger = get_logger(
    name=__name__,
    log_file="pipeline.log"
)


def main():
    logger.info(f"{STAGE_NAME} started........")

    config_manager = ConfigManager(Path("config/config.yaml"))
    evaluation_config = config_manager.get_model_evaluation_config()

    evaluation = ModelEvaluation(config=evaluation_config)
    evaluation.main_model_evaluation()

    logger.info(f"{STAGE_NAME} completed")


if __name__ == "__main__":
    main()