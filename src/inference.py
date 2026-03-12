from . import utils
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
        annotated_image = box_annotator.annotate(annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(annotated_image, detections=detections)
    sv.plot_image(annotated_image)

    annotated_image.save(out_path or "./anotated_image.jpg")
    print(f"Imagen anotada guardada como '{out_path or './anotated_image.jpg'}'")
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
    results = detect_house(imageIN, model)
    annotated_image = plot_detections(imageIN, results, out_path)
    return annotated_image