from . import utils
#from IPython.display import Image as IPyImage
from PIL import Image
import requests
from ultralytics import YOLO
import supervision as sv

model = YOLO(utils.models_path / utils.model_name)


def detect_house(image_path: str):
    try:
        imageIN = Image.open(image_path)
    except Exception as e:
        print(f"Error al abrir la imagen: {e}")
        return None
    results = model.predict(imageIN)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)
    annotated_image = imageIN.copy()
    for result in results:
        detections = sv.Detections.from_ultralytics(result)
        annotated_image = box_annotator.annotate(annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(annotated_image, detections=detections)
    sv.plot_image(annotated_image)

    annotated_image.save("./anotated_image.jpg")
    print("Imagen anotada guardada como 'annotated_image.jpg'")
    return annotated_image
