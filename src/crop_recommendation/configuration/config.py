from pathlib import Path
import os
import yaml

from crop_recommendation.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig
)


class ConfigManager:

    def __init__(self, file_path: Path):
        self.config = self._read_yaml(file_path)

    def _read_yaml(self, filepath: Path):
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found at: {filepath}")

        with open(filepath, "r") as f:
            return yaml.safe_load(f)

    def get_data_ingestion_config(self) -> DataIngestionConfig:

        ingestion = self.config.get("data_ingestion")
        if ingestion is None:
            raise ValueError("data_ingestion section missing in config.yaml")

        root_dir = Path(ingestion["root_dir"])
        os.makedirs(root_dir, exist_ok=True)

        return DataIngestionConfig(
            root_dir=root_dir,
            source_dir=Path(ingestion["source_dir"]),
            train_dir=Path(ingestion["train_dir"]),
            test_dir=Path(ingestion["test_dir"]),
        )

    def get_data_validation_config(self) -> DataValidationConfig:

        validation = self.config.get("data_validation")
        if validation is None:
            raise ValueError("data_validation section missing in config.yaml")

        root_dir = Path(validation["root_dir"])
        os.makedirs(root_dir, exist_ok=True)

        return DataValidationConfig(
            root_dir=root_dir,
            validation_status_file=Path(validation["validation_status_file"]),
            train_dir=Path(validation["train_dir"]),
            schema_file=Path(validation["schema_file"]),
        )