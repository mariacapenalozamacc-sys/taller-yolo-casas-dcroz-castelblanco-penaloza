from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from src.inference import infer, load_model


app = FastAPI(title="House Detection API", version="1.0.0")

FORMATOS_PERMITIDOS = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/webp",
}



def validar_archivo(archivo: UploadFile) -> None:
    if archivo.content_type not in FORMATOS_PERMITIDOS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {archivo.content_type}. Allowed: {sorted(FORMATOS_PERMITIDOS)}"
        )


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
async def predict(
    archivo: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold"),
    iou: float = Query(0.45, ge=0.0, le=1.0, description="IoU threshold"),
):
    validar_archivo(archivo)

    try:
        contents = await archivo.read()
        result = infer(contents, conf=conf, iou=iou, annotate=False)

        return JSONResponse(
            content={
                "filename": archivo.filename,
                "content_type": archivo.content_type,
                "total_detections": result["total_detections"],
                "detections": result["detections"],
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}") from e


@app.post("/predict/image")
async def predict_image(
    archivo: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold"),
    iou: float = Query(0.45, ge=0.0, le=1.0, description="IoU threshold"),
):
    validar_archivo(archivo)

    try:
        contents = await archivo.read()
        result = infer(contents, conf=conf, iou=iou, annotate=True)

        annotated_image = result["annotated_image"]
        #image_bytes = image_to_bytes(annotated_image, fmt="JPEG")

        return StreamingResponse(
            #BytesIO(image_bytes),
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f'attachment; filename="pred_{archivo.filename}"',
                "X-Total-Detections": str(result["total_detections"]),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}") from e