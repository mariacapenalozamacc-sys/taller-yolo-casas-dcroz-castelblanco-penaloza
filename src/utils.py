import os
from pathlib import Path
import zipfile

# Directory where this script lives
SCRIPT_DIR = Path(__file__).resolve().parent

# Project root (YOLO-AAP)
PROJECT_ROOT = SCRIPT_DIR.parent

# 
base_model_name = 'yolo11n.pt'
project_name = 'house_project'
project = "runs_house_model"
model_name = "house_yolo.pt"

# Paths
images_path = PROJECT_ROOT / "images"
models_path = PROJECT_ROOT / "models"
base_zip_path = images_path / "conf/house_project.v1i.yolov11.zip"
train_yaml_path =  images_path / "conf/data.yaml"

def unzip_dataset():
    with zipfile.ZipFile(base_zip_path, 'r') as zip_ref:
        zip_ref.extractall(images_path)


def validate_models_dir():
    if not models_path.exists():
        models_path.mkdir(parents=True, exist_ok=True)
