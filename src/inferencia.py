import utils
from ultralytics import YOLO

model = YOLO(utils.models_dir / utils.model_name)