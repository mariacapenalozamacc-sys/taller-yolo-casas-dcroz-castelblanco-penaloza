
# Proyecto YOLO - Identificación de Casas con Modelo Basado en YOLO


# Descripción general del proyecto


YOLO-house-idenfifier es una herramienta que permite el etiquetado de fachadas de casas. Puede importara libreria o mediante una API

# Estructura del repositorio

```text
taller-yolo-casas-dcroz-castelblanco-penaloza/

├── API/   # Script de validacion y clasificacion del dataset
│   ├── API_inference.py
│
├── models/
│   ├── runs_house_model/house_yolo
│   ├── __init__.py
│   ├── house_yolo.pt
│   ├── yolo11n.pt
│     
├── src/
│   ├── __init__.py
│   ├── inference.py   # Funciones
│   ├── train_yolo.py  # SCript para recrear entrenamiento
│   └── utils.py       # Funciones recurrentes y sistema de rutas
│
├── error analysis/   # Script de validacion y clasificacion del dataset
│   ├── false_positives
│   ├── false_negatives 
│
├── images/
│   ├── confg
│      ├── data.yalm 
│      ├── house_project.v1i.yolov11.zip # Etiquietas generadas desde roboflow
│   ├── test
│      ├── images
│      ├── labels
│   ├── train
│   ├── valid
│
├── README.md            # Documentación del proyecto
├── requirements.txt     # Lista de dependencias del proyecto

```


# Requerimientos

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics)


```bash
pip install ultralytics==8.4.21

pip install supervision==0.27.0.post1

pip install albumentations==2.0.8

pip install fastapi==0.135.1

pip install python_multipart==0.0.22
```
Si requiere una instalación local con uso de GPU, consultar requerimientos de [ultralytics](https://docs.ultralytics.com/quickstart/) para instalación de Torch con CUDA.

# Construcción de la Herramienta

## Datos de entrenamiento

Se obtuvieron las imágenes del siguiente [repositorio](https://drive.google.com/drive/folders/1F0ZShSpEq7DVzTN4xrlTPYH8QZA--fTg?usp=drive_link), con 69 casa donde podríamos identificar fachadas.

En las imágenes se hizo el etiquetado de la catergoría casa usando [Roboflow](https://roboflow.com/), las cuales se pueden consultar en este [proyecto](https://universe.roboflow.com/marias-workspace-grsiu/house_project-ffi14/dataset/1). Los datos se dividieron en 52 entradas para entrenamiento, 10 para validación y 7 para pruebas.

## Arquitectura del modelo

Se eligió usar un modelo [YOLO11](https://docs.ultralytics.com/es/models/yolo11/) y hacer un 
Fine Tunning para mejorar la detección fr nuestras características (casas).

## Instrucciones para reproducir el entrenamiento y la inferencia

A continuación se describen los pasos necesarios para reproducir el entrenamiento del modelo y ejecutar inferencia sobre nuevas imágenes.

### 1. Clonar el repositorio

```bash

git clone https://github.com/<usuario>/YOLO-house-identifier.git
cd YOLO-house-identifier

```

### 2. Instalar dependencias

Instalar las dependencias definidas en el archivo requirements.txt:

```bash
pip install -r requirements.txt
```

Revisar sección de requerimientos para más detalles.

### 3. Preparar el dataset (Opcional)

Puede validar que los archivos de entrenamiento esten presentes para la rutina de entrnamiento en la ruta:

images/conf/house.project.v1.yolo11.zip

La rutina de entrnamiento (train_model de src.train_yolo) ya hace la descompresión de las carpetas, no obstante, puede hacer la descompresión manual o usando la función unzip_dataset() de la liberia src.utils.

### 4. Entrenar el modelo

Para entrenar el modelo ejecutar:

python src/train_yolo.py

Este script entrena el modelo YOLO utilizando el dataset definido en data.yaml.

Los pesos generados durante el entrenamiento se almacenan en la carpeta:

models/runs_house_model/house_yolo/weights

La rutina también crea un modelo listo para importar con los pesos aplicados en la ruta 

modesl/house_yolo.pt

### 5. Evaluar el modelo

Para evaluar el desempeño del modelo y calcular métricas como falsos positivos (FP) y falsos negativos (FN), ejecutar:

python models/runs_house_model/house_yolo

Esto analiza las predicciones del modelo sobre el conjunto de validación y genera métricas de desempeño.

Si bien en la carpeta de las corridas del modelo tenemos ejemplos de las clasificaciones, y todas las métricas de evaluación y seguimiento de la entrenamiento epoca por epoca, se hace la clasificación de las imágenes según la matriz de confusión para tener la totalidad de ejemplo de FP y FN.

### 5. Usar el modelo

El script de inferencia cargará el modelo entrenado y generará detecciones sobre las imágenes de prueba, dibujando los bounding boxes correspondientes. Se puede emplear de 3 modos:

### Mediante liberia (Python)

Estructura

```python
from src.inference import infer
res = inf.infer(image_path = './input_path/image.jpg' )
```

**Ejemplo**
```python
from src import inference as inf
res = inf.infer(image_path = './images/valid/images/real_041_MI_HOUSE_png.rf.0e452c2b0051f1281e7c048c3f3d5605.jpg'
                ,out_path= './examples' )
```


### Mediante terminal (CLI)

```bash
python src/inference.py ./input_path/image.jpg --output./output_path

```

**Ejemplo**
```bash
python ./src/inference.py images\valid\images\real_011_Casa_de_Cundinamarca_png.rf.b1da1d02fbd326e25e036adc2c977503.jpg -o./examples
```

### Mediante API

Para realizar detección de casas sobre imágenes nuevas ejecutar:






---

# Resultados (métricas) y ejemplos de detección

## Resultados (métricas) y ejemplos de detección

El modelo fue evaluado utilizando el conjunto de validación definido en el dataset.

Las métricas principales utilizadas fueron:

- **Precision**
- **Recall**
- **Falsos Positivos (FP)**
- **Falsos Negativos (FN)**

Resultados obtenidos:

| Métrica | Valor |
|------|------|
| Precision | 0.8571 |
| Recall |  0.7500 |
| False Positives | 2 |
| False Negatives | 4 |

### Ejemplo de detección correcta

En la siguiente imágen, el modelo logra identificar correctamente la fachadas de una casa presente en la escena.

![Alt text](./examples/real_041_MI_HOUSE_png.rf.0e452c2b0051f1281e7c048c3f3d5605.jpg)

### Ejemplos de errores de detección

Se identificaron algunos casos donde el modelo presenta errores y los guarda en la carpeta error_analysis.

#### **Falsos positivos (FP)**  
El modelo detecta una casa en objetos visualmente similares, como edificios o estructuras arquitectónicas.

**Ejemplo**

 En rojo se muestran las predicciones, y en verde las etiquetas.

![Alt text](./error_analysis/false_positives/real_049_Providencia_Colombia_-_panoramio_29__png.rf.c52bfec95485cfc75f327cb31ee0c41e.jpg)

**Falsos negativos (FN)**  
El modelo no detecta casas cuando:

- la fachada está parcialmente oculta
- la iluminación es baja
- la casa aparece muy pequeña en la imagen

**Ejemplo**

 En verde se muestran las predicciones, y en rojo las etiquetas.

![Alt text](./error_analysis/false_negatives/real_076_Kogisiedlung_png.rf.511c7d5c9b0d54fc0a922aa31e4beb32.jpg)

# Limitaciones y pasos futuros recomendados

### Limitaciones del modelo

A pesar de los resultados obtenidos, el modelo presenta algunas limitaciones:

1. **Tamaño reducido del dataset**

El modelo fue entrenado con un conjunto de aproximadamente 69 imágenes, lo cual es un tamaño limitado para entrenar modelos de detección de objetos robustos.

Esto puede provocar:

- sobreajuste (overfitting)
- baja capacidad de generalización a nuevas imágenes.

2. **Variabilidad limitada en las escenas**

Las imágenes del dataset no cubren completamente todas las variaciones posibles de:

- arquitectura
- iluminación
- ángulos de cámara
- contextos urbanos y rurales.

3. **Confusión con estructuras similares**

El modelo puede confundir casas con:

- edificios
- locales comerciales
- construcciones con fachada similar.
- Reflejos en agua de la misma casa

4. **Resolución de imagen**

En imágenes donde la casa aparece muy pequeña, el modelo presenta dificultades para detectar correctamente el objeto.

---

### Trabajo futuro

Para mejorar el desempeño del modelo se recomiendan las siguientes acciones:

**1. Ampliar el dataset**

Recolectar y etiquetar más imágenes de casas en diferentes contextos:

- urbano
- rural
- diferentes regiones de Colombia
- diferentes condiciones de iluminación.

Idealmente aumentar el dataset a **200–500 imágenes**.

**2. Aplicar técnicas de aumento de datos**

Utilizar técnicas de *data augmentation* como:

- rotaciones
- cambios de brillo y contraste
- escalamiento
- transformaciones geométricas.

**3. Ajuste de hiperparámetros**

Realizar experimentación con diferentes valores de:

- número de épocas
- tamaño de imagen
- learning rate
- batch size.

**4. Evaluación con más métricas**

Incorporar análisis adicionales como:

- matriz de confusión
- curvas Precision-Recall
- evaluación en datasets externos.












