from pathlib import Path
import os 
import yaml
from crop_recommendation.entity.config_entity import (
    DataIngestionConfig
)

class ConfigManager : 
    def __init__(self,file_path : Path):
        self.config=self._read_yaml(file_path)

    def _read_yaml(self,filepath : Path):
        with open(filepath,'r') as f:
            return yaml.safe_load(f)
        
    def get_data_ingestion_config(self)->DataIngestionConfig:

        ingestion = self.config["data_ingestion"]
        os.makedirs(ingestion['root_dir'],exist_ok=True)

        return DataIngestionConfig(
            root_dir=Path(ingestion['root_dir']),
            source_dir=Path(ingestion['source_dir']),
            train_dir=Path(ingestion['train_dir']),
            test_dir=Path(ingestion['test_dir'])
        )