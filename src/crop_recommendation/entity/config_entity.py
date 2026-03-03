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
    
@dataclass
class DataPreprocessingConfig:
    root_dir: Path
    train_dir: Path
    test_dir: Path
    processed_train_dir: Path
    processed_test_dir: Path
    scaler_path: Path

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    processed_train_dir: Path
    processed_test_dir: Path
    model_path: Path
    params_file: Path
