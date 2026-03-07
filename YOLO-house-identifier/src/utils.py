import os
from pathlib import Path

# Directory where this script lives
SCRIPT_DIR = Path(__file__).resolve().parent

# Project root (YOLO-AAP)
PROJECT_ROOT = SCRIPT_DIR.parent

# 
base_model_name = 'yolo11n.pt'
project_name = 'house_project'
project = "runs_house"
model_name = "house_yolo"

# Paths
images_path = PROJECT_ROOT / "images"
base_zip_path = PROJECT_ROOT / "house_project.v1i.yolov11.zip"
models_dir = PROJECT_ROOT / "models"
train_yaml_path =  PROJECT_ROOT / "images" / "data.yaml"

def unzip_dataset():
    os.system(f'unzip -o "{base_zip_path}" -d "{images_path}"')

def validate_models_dir():
    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
