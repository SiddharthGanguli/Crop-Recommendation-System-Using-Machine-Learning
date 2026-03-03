from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_dir: Path
    train_dir: Path
    test_dir: Path


@dataclass
class DataValidationConfig:
    root_dir: Path
    validation_status_file: Path
    train_dir: Path
    schema_file: Path