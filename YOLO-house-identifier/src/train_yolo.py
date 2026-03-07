import os
from pathlib import Path
import utils
from ultralytics import YOLO
import albumentations as A

utils.unzip_dataset()
utils.validate_models_dir()

def train(data_yaml: str, epochs: int = 50, imgsz: int = 640, batch: int = 8):

    # Define base model from trasfer learning
    model = YOLO(utils.base_model_name)

    # Prepare directories
    utils.validate_models_dir()

    # Data augmentation
    custom_transforms = [
      A.Blur(blur_limit=7, p=0.5),
      A.CLAHE(clip_limit=4.0, p=0.5),
    ]

    # Train model
    results = model.train(
        data = data_yaml
        ,epochs = epochs
        ,imgsz = imgsz
        ,batch = batch
        ,project = utils.project
        ,name = utils.model_name
        ,exist_ok = True
        ,augmentations = custom_transforms
        ,verbose = False
    )

    # Save models
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if best_weights.exists():
        target = utils.models_dir / utils.model_name
        target.write_bytes(best_weights.read_bytes())
        print(f"Pesos guardados en {target}")

    return results