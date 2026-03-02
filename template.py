import os
import logging
from pathlib import Path

project_name = "crop_recommendation"

log_dir = "logs"
log_file = "project_setup.log"

os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, log_file),
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    encoding="utf-8"
)

logger = logging.getLogger(__name__)

files = [

    f"src/{project_name}/__init__.py",

    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",

    f"src/{project_name}/configuration/__init__.py",
    f"src/{project_name}/configuration/config.py",

    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/data_preprocessing.py",
    f"src/{project_name}/components/training.py",
    f"src/{project_name}/components/evaluation.py",

    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/01_data_ingestion_pipeline.py",
    f"src/{project_name}/pipeline/02_data_validation_pipeline.py",
    f"src/{project_name}/pipeline/03_data_preprocessing_pipeline.py",
    f"src/{project_name}/pipeline/04_training_pipeline.py",
    f"src/{project_name}/pipeline/05_evaluation_pipeline.py",

    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",

    "data/raw/.gitkeep",
    "data/processed/.gitkeep",

    "artifacts/.gitkeep",

    "notebooks/.gitkeep",

    "tests/__init__.py",
    "tests/test_training.py",

    "config/config.yaml",
    "params.yaml",

    "dvc.yaml",
    ".dvc/config",

    "main.py",
    "app.py",

    "Dockerfile",
    "docker-compose.yaml",
    ".dockerignore",

    ".github/workflows/ci.yaml",

    "requirements.txt",
    "setup.py",
    ".gitignore",
    "README.md"
]



def create_files(file_list):
    for file_path in file_list:
        file_path = Path(file_path)
        file_dir = file_path.parent

        if file_dir and not file_dir.exists():
            file_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"{file_dir} directory created")

        if not file_path.exists():
            file_path.touch()
            logger.info(f"{file_path} file created")
        else:
            logger.info(f"{file_path} already exists")

if __name__ == "__main__":
    logger.info("Project structure creation started.")
    create_files(files)
    logger.info("Project structure creation completed.")