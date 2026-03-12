from . import inference as inf
from . import utils
import pandas as pd

valid_images_dir = utils.images_path / 'valid/images'
valid_labels_dir = utils.images_path / 'valid/labels'
utils.unzip_dataset()


model = inf.load_model()

from pathlib import Path
from PIL import Image
import numpy as np


def yolo_to_xyxy(x_center, y_center, width, height, img_w, img_h):
    """
    Convierte una caja YOLO normalizada (xc, yc, w, h) a formato xyxy absoluto.
    """
    xc = x_center * img_w
    yc = y_center * img_h
    w = width * img_w
    h = height * img_h

    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2

    x1 = max(0, min(x1, img_w))
    y1 = max(0, min(y1, img_h))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))

    return (x1, y1, x2, y2)


def load_ground_truth_yolo(label_path, image_path):
    """
    Lee un archivo .txt en formato YOLO y devuelve las cajas GT
    en formato absoluto xyxy.
    """
    label_path = Path(label_path)
    image_path = Path(image_path)

    if not label_path.exists():
        return []

    if label_path.suffix.lower() != ".txt":
        raise ValueError(f"Se esperaba un archivo .txt y llegó: {label_path}")

    image = Image.open(image_path)
    img_w, img_h = image.size

    gt_boxes = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            parts = line.strip().split()

            if not parts:
                continue

            if len(parts) != 5:
                raise ValueError(
                    f"Línea inválida en {label_path}, línea {line_num}: {line!r}"
                )

            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])

            box_xyxy = yolo_to_xyxy(
                x_center, y_center, width, height, img_w, img_h
            )

            gt_boxes.append({
                "class_id": class_id,
                "box_xyxy": box_xyxy
            })

    return gt_boxes


def compute_iou(box_a, box_b):
    """
    Calcula IoU entre dos cajas en formato xyxy.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def extract_predictions_from_ultralytics(result):
    """
    Extrae predicciones desde result de Ultralytics.
    """
    preds = []

    if result.boxes is None or len(result.boxes) == 0:
        return preds

    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)

    for box, score, class_id in zip(xyxy, conf, cls):
        preds.append({
            "class_id": int(class_id),
            "confidence": float(score),
            "box_xyxy": tuple(box.tolist())
        })

    return preds


def build_iou_matrix(predictions, ground_truths):
    """
    Construye la matriz IoU entre predicciones y GT.
    """
    iou_matrix = np.zeros((len(predictions), len(ground_truths)), dtype=float)

    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truths):
            if pred["class_id"] == gt["class_id"]:
                iou_matrix[i, j] = compute_iou(pred["box_xyxy"], gt["box_xyxy"])

    return iou_matrix


def classify_image_detections(predictions, ground_truths, iou_threshold=0.5):
    """
    Clasifica predicciones y GT de una imagen en TP, FP, FN
    usando matching greedy por IoU.
    """
    iou_matrix = build_iou_matrix(predictions, ground_truths)

    if len(predictions) == 0 and len(ground_truths) == 0:
        return {
            "tp_matches": [],
            "fp_predictions": [],
            "fn_ground_truths": [],
            "iou_matrix": iou_matrix,
        }

    if len(predictions) == 0:
        return {
            "tp_matches": [],
            "fp_predictions": [],
            "fn_ground_truths": [
                {"gt_idx": j, "ground_truth": ground_truths[j]}
                for j in range(len(ground_truths))
            ],
            "iou_matrix": iou_matrix,
        }

    if len(ground_truths) == 0:
        return {
            "tp_matches": [],
            "fp_predictions": [
                {"pred_idx": i, "prediction": predictions[i]}
                for i in range(len(predictions))
            ],
            "fn_ground_truths": [],
            "iou_matrix": iou_matrix,
        }

    candidates = []
    for i in range(len(predictions)):
        for j in range(len(ground_truths)):
            iou_val = iou_matrix[i, j]
            if iou_val >= iou_threshold:
                candidates.append((i, j, float(iou_val)))

    candidates.sort(key=lambda x: x[2], reverse=True)

    matched_pred = set()
    matched_gt = set()
    tp_matches = []

    for pred_idx, gt_idx, iou_val in candidates:
        if pred_idx not in matched_pred and gt_idx not in matched_gt:
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)
            tp_matches.append({
                "pred_idx": pred_idx,
                "gt_idx": gt_idx,
                "iou": iou_val,
                "prediction": predictions[pred_idx],
                "ground_truth": ground_truths[gt_idx],
            })

    fp_predictions = []
    for i in range(len(predictions)):
        if i not in matched_pred:
            fp_predictions.append({
                "pred_idx": i,
                "prediction": predictions[i],
            })

    fn_ground_truths = []
    for j in range(len(ground_truths)):
        if j not in matched_gt:
            fn_ground_truths.append({
                "gt_idx": j,
                "ground_truth": ground_truths[j],
            })

    return {
        "tp_matches": tp_matches,
        "fp_predictions": fp_predictions,
        "fn_ground_truths": fn_ground_truths,
        "iou_matrix": iou_matrix,
    }

print("Funciones cargadas correctamente.")

image_paths = sorted(
    list(valid_images_dir.glob("*.jpg")) +
    list(valid_images_dir.glob("*.jpeg")) +
    list(valid_images_dir.glob("*.png"))
)

if len(image_paths) == 0:
    raise FileNotFoundError("No se encontraron imágenes en valid/images")

all_results = []
global_tp = 0
global_fp = 0
global_fn = 0

IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.25

for image_path in image_paths:
    label_path = valid_labels_dir / f"{image_path.stem}.txt"

    # Si no existe label, asumimos que no hay GT para esa imagen
    if label_path.exists():
        ground_truths = load_ground_truth_yolo(label_path, image_path)
    else:
        ground_truths = []

    # Predicción del modelo
    results = model.predict(str(image_path), conf=CONF_THRESHOLD, verbose=False)
    predictions = extract_predictions_from_ultralytics(results[0])

    # Clasificación TP/FP/FN para esta imagen
    classified = classify_image_detections(
        predictions=predictions,
        ground_truths=ground_truths,
        iou_threshold=IOU_THRESHOLD
    )

    tp_count = len(classified["tp_matches"])
    fp_count = len(classified["fp_predictions"])
    fn_count = len(classified["fn_ground_truths"])

    global_tp += tp_count
    global_fp += fp_count
    global_fn += fn_count

    all_results.append({
        "image_path": str(image_path),
        "label_path": str(label_path),
        "num_predictions": len(predictions),
        "num_ground_truths": len(ground_truths),
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "tp_matches": classified["tp_matches"],
        "fp_predictions": classified["fp_predictions"],
        "fn_ground_truths": classified["fn_ground_truths"],
        "iou_matrix": classified["iou_matrix"],
    })

print("=== RESUMEN GLOBAL ===")
print(f"Imágenes evaluadas: {len(all_results)}")
print(f"TP totales: {global_tp}")
print(f"FP totales: {global_fp}")
print(f"FN totales: {global_fn}")

summary_rows = []

for r in all_results:
    summary_rows.append({
        "image_name": Path(r["image_path"]).name,
        "num_predictions": r["num_predictions"],
        "num_ground_truths": r["num_ground_truths"],
        "tp": r["tp"],
        "fp": r["fp"],
        "fn": r["fn"],
    })

summary_df = pd.DataFrame(summary_rows)

summary_df = summary_df.sort_values(
    by=["fn", "fp", "tp"],
    ascending=[False, False, True]
).reset_index(drop=True)

precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
f1_score = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0 else 0.0
)

print("=== MÉTRICAS GLOBALES ===")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1_score:.4f}")

#Tabla TP

tp_rows = []

for r in all_results:
    image_name = Path(r["image_path"]).name
    for m in r["tp_matches"]:
        tp_rows.append({
            "image_name": image_name,
            "pred_idx": m["pred_idx"],
            "gt_idx": m["gt_idx"],
            "iou": m["iou"],
            "confidence": m["prediction"]["confidence"],
            "pred_box": m["prediction"]["box_xyxy"],
            "gt_box": m["ground_truth"]["box_xyxy"],
        })

tp_df = pd.DataFrame(tp_rows)
tp_df.head()

#Tabla FP

fp_rows = []

for r in all_results:
    image_name = Path(r["image_path"]).name
    for fp in r["fp_predictions"]:
        fp_rows.append({
            "image_name": image_name,
            "pred_idx": fp["pred_idx"],
            "confidence": fp["prediction"]["confidence"],
            "pred_box": fp["prediction"]["box_xyxy"],
        })

fp_df = pd.DataFrame(fp_rows)
fp_df.head()

#Tabla FN

fn_rows = []

for r in all_results:
    image_name = Path(r["image_path"]).name
    for fn in r["fn_ground_truths"]:
        fn_rows.append({
            "image_name": image_name,
            "gt_idx": fn["gt_idx"],
            "gt_box": fn["ground_truth"]["box_xyxy"],
        })

fn_df = pd.DataFrame(fn_rows)
fn_df.head()

from pathlib import Path
from PIL import Image, ImageDraw

OUTPUT_DIR = Path("error_analysis")
FP_DIR = OUTPUT_DIR / "false_positives"
FN_DIR = OUTPUT_DIR / "false_negatives"

FP_DIR.mkdir(parents=True, exist_ok=True)
FN_DIR.mkdir(parents=True, exist_ok=True)

print("Carpetas creadas:")
print(FP_DIR)
print(FN_DIR)

def draw_boxes_on_image(image_path, gt_boxes=None, pred_boxes=None,
                        gt_color="green", pred_color="red", line_width=3):
    """
    Dibuja cajas ground truth y predicción sobre una imagen.

    Parámetros
    ----------
    image_path : str o Path
        Ruta de la imagen.
    gt_boxes : list[tuple]
        Lista de cajas GT en formato (x1, y1, x2, y2).
    pred_boxes : list[tuple]
        Lista de cajas predichas en formato (x1, y1, x2, y2).

    Retorna
    -------
    PIL.Image
        Imagen anotada.
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    gt_boxes = gt_boxes or []
    pred_boxes = pred_boxes or []

    for box in gt_boxes:
        draw.rectangle(box, outline=gt_color, width=line_width)

    for box in pred_boxes:
        draw.rectangle(box, outline=pred_color, width=line_width)

    return image

saved_fp_images = []

for r in all_results:
    if len(r["fp_predictions"]) == 0:
        continue

    image_path = Path(r["image_path"])

    # cajas GT de la imagen
    gt_boxes = []
    for tp in r["tp_matches"]:
        gt_boxes.append(tp["ground_truth"]["box_xyxy"])
    for fn in r["fn_ground_truths"]:
        gt_boxes.append(fn["ground_truth"]["box_xyxy"])

    # también puede haber GT no repetidos si quieres reconstruirlos desde ground_truths,
    # pero con tp_matches + fn_ground_truths normalmente cubres todos los GT usados/no usados

    # cajas FP
    fp_boxes = [fp["prediction"]["box_xyxy"] for fp in r["fp_predictions"]]

    annotated_image = draw_boxes_on_image(
        image_path=image_path,
        gt_boxes=gt_boxes,
        pred_boxes=fp_boxes,
        gt_color="green",
        pred_color="red",
        line_width=3,
    )

    output_path = FP_DIR / image_path.name
    annotated_image.save(output_path)
    saved_fp_images.append(str(output_path))

print(f"Se guardaron {len(saved_fp_images)} imágenes con FP en {FP_DIR}")

saved_fn_images = []

for r in all_results:
    if len(r["fn_ground_truths"]) == 0:
        continue

    image_path = Path(r["image_path"])

    # cajas GT que quedaron sin detectar = falsos negativos
    fn_gt_boxes = [fn["ground_truth"]["box_xyxy"] for fn in r["fn_ground_truths"]]

    # predicciones presentes en esa imagen (TP + FP), para contexto visual
    pred_boxes = [tp["prediction"]["box_xyxy"] for tp in r["tp_matches"]]
    pred_boxes += [fp["prediction"]["box_xyxy"] for fp in r["fp_predictions"]]

    annotated_image = draw_boxes_on_image(
        image_path=image_path,
        gt_boxes=fn_gt_boxes,
        pred_boxes=pred_boxes,
        gt_color="green",
        pred_color="red",
        line_width=3,
    )

    output_path = FN_DIR / image_path.name
    annotated_image.save(output_path)
    saved_fn_images.append(str(output_path))

print(f"Se guardaron {len(saved_fn_images)} imágenes con FN en {FN_DIR}")