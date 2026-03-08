import os
from pathlib import Path
from . import utils
from ultralytics import YOLO
import albumentations as A



def train(data_yaml: str, epochs: int = 50, imgsz: int = 640, batch: int = 8):

    # Define base model from trasfer learning
    model = YOLO(utils.models_path / utils.base_model_name)

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
        ,project = utils.models_path / utils.project
        ,name = utils.model_name
        ,exist_ok = True
        ,augmentations = custom_transforms
        ,verbose = False
    )

    # Save models
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if best_weights.exists():
        target = utils.models_path / utils.model_name
        target.write_bytes(best_weights.read_bytes())
        print(f"Pesos guardados en {target}")

    return results


def train_model(data_yaml: str = utils.train_yaml_path, epochs: int = 50, imgsz: int = 640, batch: int = 8):
    
    
    utils.unzip_dataset()
    print("Dataset unzipped.")
    utils.validate_models_dir()
    print("models directory validated.")
    results = train(data_yaml, epochs, imgsz, batch)
    return results

if __name__ == "__main__":
    _ = train_model()