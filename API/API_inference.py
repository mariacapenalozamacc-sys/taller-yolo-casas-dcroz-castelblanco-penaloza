from pathlib import Path
import sys
from io import BytesIO
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import infer, load_model, load_image, detect_house


app = FastAPI(title="House Detection API", version="1.0.0")

FORMATOS_PERMITIDOS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/bmp": ".bmp",
    "image/webp": ".webp",
}

MODEL = None


@app.on_event("startup")
def startup_event():
    global MODEL
    MODEL = load_model()


def validar_archivo(archivo: UploadFile) -> str:
    if archivo.content_type not in FORMATOS_PERMITIDOS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {archivo.content_type}. Allowed: {sorted(FORMATOS_PERMITIDOS)}"
        )
    return FORMATOS_PERMITIDOS[archivo.content_type]


def serialize_results(results):
    detections_out = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for xyxy, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            cls_id = int(cls_id.item())
            conf = float(conf.item())
            coords = [float(v) for v in xyxy.tolist()]

            detections_out.append(
                {
                    "class_id": cls_id,
                    "class_name": result.names[cls_id],
                    "confidence": round(conf, 4),
                    "bbox_xyxy": [round(v, 2) for v in coords],
                }
            )

    return detections_out


@app.get("/")
def raiz():
    return {
        "api": "House Detection API",
        "endpoints": {
            "POST /predict": "Return JSON detections",
            "POST /predict/image": "Return annotated image",
        },
    }


@app.post("/predict")
async def predict(archivo: UploadFile = File(...)):
    extension = validar_archivo(archivo)

    try:
        contents = await archivo.read()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            input_path = tmp_dir / f"input{extension}"
            input_path.write_bytes(contents)

            image_in = load_image(str(input_path))
            if image_in is None:
                raise HTTPException(status_code=400, detail="Invalid image file")

            model_to_use = MODEL if MODEL is not None else load_model()
            results = detect_house(image_in, model=model_to_use)
            detections = serialize_results(results)

            return JSONResponse(
                content={
                    "filename": archivo.filename,
                    "content_type": archivo.content_type,
                    "total_detections": len(detections),
                    "detections": detections,
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}") from e


@app.post("/predict/image")
async def predict_image(archivo: UploadFile = File(...)):
    extension = validar_archivo(archivo)

    try:
        contents = await archivo.read()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            input_path = tmp_dir / f"input{extension}"
            output_path = tmp_dir / f"pred_{Path(archivo.filename).name}"

            input_path.write_bytes(contents)

            model_to_use = MODEL if MODEL is not None else load_model()

            annotated_image = infer(
                image_path=str(input_path),
                model=model_to_use,
                out_path=str(output_path),
            )

            if annotated_image is None:
                raise HTTPException(status_code=500, detail="Inference returned None")

            if not output_path.exists():
                raise HTTPException(status_code=500, detail="Annotated image was not created")

            image_bytes = output_path.read_bytes()

            return StreamingResponse(
                BytesIO(image_bytes),
                media_type="image/jpeg",
                headers={
                    "Content-Disposition": f'attachment; filename="{output_path.name}"',
                },
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}") from e