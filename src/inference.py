import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils
#from IPython.display import Image as IPyImage
from PIL import Image
from ultralytics import YOLO
import supervision as sv



def load_model():
    model = YOLO(utils.models_path / utils.model_name)
    # model = (YOLO(utils.models_path / utils.base_model_name)
    #         .load(utils.best_weights_path))
    return model    

def load_image(image_path: str):
    try:
        imageIN = Image.open(image_path)
        return imageIN
    except Exception as e:
        print(f"Error al abrir la imagen: {e}")
        return None

def detect_house(imageIN, model = None):
    if model is None:
        model = load_model()
    results = model.predict(imageIN)
    return results


def plot_detections(imageIN, results, out_path : str = None):
    '''
        Anota las detecciones en la imagen y la guarda.
        Args:
            imageIN (PIL.Image): Imagen de entrada.
            results: Resultados de la detección del modelo YOLO.
            out_path (str, optional): Ruta para guardar la imagen anotada. Si no se proporciona, se guardará como 'annotated_image.jpg' en el directorio actual.
        Returns:
            annotated_image (PIL.Image): Imagen anotada con las detecciones.
    '''
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)

    
    annotated_image = imageIN.copy()
    for result in results:
        detections = sv.Detections.from_ultralytics(result)
         # Build labels with class name and probability
        labels = [
            f"{result.names[int(class_id)]} {confidence:.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]

        annotated_image = box_annotator.annotate(
            annotated_image,
            detections=detections
        )

        annotated_image = label_annotator.annotate(
            annotated_image,
            detections=detections,
            labels=labels
        )
    sv.plot_image(annotated_image)

    if out_path:
        annotated_image.save(out_path)
        print(f"Imagen anotada guardada como '{out_path}'")
    return annotated_image

def infer(image_path : str, model = None,out_path : str = None):
    '''
        Realiza la inferencia en una imagen dada y guarda la imagen anotada.
        Args:
            image_path (str): Ruta de la imagen de entrada.
            model (YOLO, optional): Modelo YOLO cargado. Si no se proporciona, se cargará el modelo por defecto.
            out_path (str, optional): Ruta para guardar la imagen anotada. Si no se proporciona, se guardará como 'annotated_image.jpg' en el directorio actual.
        Returns:
            annotated_image (PIL.Image): Imagen anotada con las detecciones.
    '''
    imageIN = load_image(image_path)
    if imageIN is None:
        return None
    
    image_path = Path(image_path)
    original_name = image_path.name

    # determine output location
    if out_path is None:
        save_path = Path.cwd() / original_name

    else:
        out_path = Path(out_path)

        if out_path.is_dir():
            save_path = out_path / original_name
        else:
            save_path = out_path
    results = detect_house(imageIN, model)
    annotated_image = plot_detections(imageIN, results, save_path)
    return annotated_image

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Run YOLO house detection inference"
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to the input image"
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output directory or file path"
    )

    args = parser.parse_args()

    model = load_model()

    infer(
        image_path=args.image,
        model=model,
        out_path=args.output
    )