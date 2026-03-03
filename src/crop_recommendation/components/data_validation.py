import os
from pathlib import Path
import yaml
import pandas as pd

from crop_recommendation.entity.config_entity import DataValidationConfig
from crop_recommendation.utils.logger import get_logger

logger = get_logger(
    name=__name__,
    log_file="data_validation.log"
)


class DataValidation:

    def __init__(self, config: DataValidationConfig):
        self.config = config

    def _read_schema(self):
        if not self.config.schema_file.exists():
            raise FileNotFoundError(f"Schema file not found at {self.config.schema_file}")

        with open(self.config.schema_file, "r") as f:
            return yaml.safe_load(f)

    def _validate_columns(self, df: pd.DataFrame, expected_cols: list):
        actual_columns = list(df.columns)

        if set(expected_cols) != set(actual_columns):
            logger.error("Column names mismatch between schema and dataset.")
            return False

        logger.info("Column validation passed.")
        return True

    def run(self):
        try:
            logger.info("Data validation started.")

            if not self.config.train_dir.exists():
                raise FileNotFoundError(f"Train file not found at {self.config.train_dir}")

            schema = self._read_schema()
            expected_cols = list(schema["columns"].keys())

            df = pd.read_csv(self.config.train_dir)

            status = self._validate_columns(df, expected_cols)

            os.makedirs(self.config.root_dir, exist_ok=True)

            with open(self.config.validation_status_file, "w") as f:
                f.write(str(status))

            logger.info(f"Validation status: {status}")
            logger.info("Data validation completed.")

            return status

        except Exception as e:
            logger.exception("Error during data validation.")
            raise e