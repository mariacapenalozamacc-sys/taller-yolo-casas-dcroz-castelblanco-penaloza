# Detección de Casas en Imágenes Colombianas usando YOLO

Proyecto para detección de fachadas de casas en imágenes usando un modelo basado en YOLO.  
El repositorio permite:

- ejecutar inferencia sobre imágenes individuales,
- usar el modelo desde Python como librería,
- exponer el modelo mediante una API con FastAPI,
- reentrenar el modelo,
- y revisar resultados de validación / análisis de errores.

---

## 1. Estructura del repositorio

```text
taller-yolo-casas-dcroz-castelblanco-penaloza/
├── API/
│   ├── API_inference.py
│   └── __init__.py
│
├── error_analysis/
│   └── ...
│
├── examples/
│   ├── API_tutorial_01.png
│   ├── API_tutorial_02.png
│   ├── API_tutorial_03.png
│   ├── API_tutorial_04.png
│   └── imágenes de ejemplo
│
├── images/
│   ├── conf/
│   └── data.yaml
│
├── models/
│   ├── runs_house_model/ house_yolo
│   ├── __init__.py
│   ├── house_yolo.pt
│   └── yolo11n.pt
│
├── src/
│   ├── __init__.py
│   ├── inference.py
│   ├── train_yolo.py
│   ├── utils.py
│   └── validation.py
│
├── README.md
└── requirements.txt
```

> Nota: la carpeta `images/` visible en el repositorio contiene `conf/` y `data.yaml`.  
> Si el entrenamiento o la validación requieren `train/`, `valid/` y `test/`, estos deben generarse o descomprimirse a partir del dataset configurado por el proyecto.

---

## 2. Requisitos

Se recomienda usar **Python 3.10+**.

Dependencias principales del proyecto:

- `ultralytics==8.4.21`
- `supervision==0.27.0.post1`
- `albumentations==2.0.8`
- `fastapi==0.135.1`
- `python_multipart==0.0.22`
- `uvicorn==0.24.0`

Instalación:

```bash
pip install -r requirements.txt
```

Si vas a entrenar con GPU, instala también la versión de `torch` compatible con tu CUDA según la documentación oficial de Ultralytics/PyTorch.

---

## 3. Clonar el repositorio

```bash
git clone https://github.com/mariacapenalozamacc-sys/taller-yolo-casas-dcroz-castelblanco-penaloza.git
cd taller-yolo-casas-dcroz-castelblanco-penaloza
```

---

## 4. Preparación de datos

El proyecto usa un archivo de configuración de datos en:

```text
images/data.yaml
```

Además, la carpeta `images/conf/` contiene archivos de configuración asociados al dataset.

Dependiendo de cómo esté implementado `utils.py`, el flujo de entrenamiento/validación puede descomprimir automáticamente el dataset.  
Aun así, antes de entrenar debes verificar que existan las carpetas esperadas por YOLO, por ejemplo:

```text
images/train/images
images/train/labels
images/valid/images
images/valid/labels
images/test/images
images/test/labels
```

Si esas carpetas no existen todavía, revisa la lógica de preparación de datos en `src/utils.py` y el dataset comprimido asociado al proyecto.

---

## 5. Uso como librería

Puedes correr inferencia importando el módulo desde Python.

### Ejemplo

```python
from src.inference import load_model, infer

model = load_model()
infer(
    image_path="examples/real_011_Casa_de_Cundinamarca_png.rf.b1da1d02fbd326e25e036adc2c977503.jpg",
    model=model,
    out_path="outputs/"
)
```

Esto carga el modelo configurado por defecto y guarda la imagen anotada en la carpeta `outputs/`.

---

## 6. Uso desde línea de comandos

El script `src/inference.py` expone una interfaz CLI.

### Sintaxis

```bash
python src/inference.py <ruta_imagen> --output <ruta_salida>
```

### Ejemplo

```bash
python src/inference.py examples/real_011_Casa_de_Cundinamarca_png.rf.b1da1d02fbd326e25e036adc2c977503.jpg --output outputs/
```

También puedes usar la forma corta:

```bash
python src/inference.py examples/real_011_Casa_de_Cundinamarca_png.rf.b1da1d02fbd326e25e036adc2c977503.jpg -o outputs/
```

Si `--output` apunta a un directorio, el script guarda la imagen anotada allí con el mismo nombre original.  
Si `--output` apunta a un archivo, se guarda exactamente en esa ruta.

---

## 7. Entrenamiento

El script de entrenamiento del repositorio es:

```text
src/train_yolo.py
```

Para lanzar el entrenamiento:

```bash
python src/train_yolo.py
```

De acuerdo con la implementación actual, el script:

- carga el modelo base desde `models/`,
- prepara / valida directorios,
- intenta descomprimir el dataset,
- entrena el modelo,
- y copia los mejores pesos entrenados a la ruta del modelo final.

Si necesitas cambiar hiperparámetros como `epochs`, `imgsz` o `batch`, puedes modificar la función `train()` o `train_model()` dentro de `src/train_yolo.py`.

---

## 8. Validación y análisis de errores

El proyecto incluye el script:

```text
src/validation.py
```

Este módulo implementa lógica para:

- leer etiquetas en formato YOLO,
- convertir bounding boxes a formato absoluto,
- calcular IoU,
- extraer predicciones del resultado de Ultralytics,
- clasificar detecciones en TP / FP / FN,
- y apoyar el análisis visual de errores.

Según la estructura del repositorio, los artefactos relacionados con errores se almacenan en:

```text
error_analysis/
```

Esto permite revisar falsos positivos y falsos negativos para entender mejor el comportamiento del modelo.

---

## 9. API con FastAPI

La API está implementada en:

```text
API/API_inference.py
```

### Levantar el servidor

```bash
uvicorn API.API_inference:app --reload
```

### Endpoints principales

#### `GET /`
Devuelve información general de la API.

#### `POST /predict`
Recibe una imagen y retorna un JSON con las detecciones.

#### `POST /predict/image`
Recibe una imagen y devuelve la imagen anotada con bounding boxes.

---

## 10. Ejemplo de consumo de la API

Una vez el servidor esté corriendo, puedes probar el endpoint `/predict` desde Swagger UI o con `curl`.

### Ejemplo con `curl`

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "archivo=@examples/real_011_Casa_de_Cundinamarca_png.rf.b1da1d02fbd326e25e036adc2c977503.jpg"
```

### Ejemplo para obtener imagen anotada

```bash
curl -X POST "http://127.0.0.1:8000/predict/image" \
  -H "accept: image/jpeg" \
  -H "Content-Type: multipart/form-data" \
  -F "archivo=@examples/real_011_Casa_de_Cundinamarca_png.rf.b1da1d02fbd326e25e036adc2c977503.jpg" \
  --output prediccion.jpg
```

---

## 11. Modelos

La carpeta `models/` contiene:

- `yolo11n.pt`: modelo base,
- `house_yolo.pt`: pesos del modelo entrenado,
- historial / artefactos de entrenamiento en `runs_house_model/ house_yolo`.

El modelo de inferencia usa por defecto los pesos configurados en `src/utils.py`.

---

## 12.  Autores

Repositorio desarrollado para el taller de detección de casas con YOLO por Miguel CastelBlanco, ANthony D'croz y Maria Camila Peñaloza.
