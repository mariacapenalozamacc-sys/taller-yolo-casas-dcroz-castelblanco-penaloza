
# Proyecto YOLO - Identificación de Casas con Modelo Basado en YOLO


# Descripción general del proyecto


YOLO-house-idenfifier es una herramienta que permite el etiquetado de fachadas de casas. Puede importara libreria o mediante una API

# Estructura del repositorio

```text
YOLO-house-identifier/
│
├── examples/            
│
├── src/
│   ├── inference.py   # Funciones para 
│   ├── train_yolo.py  # SCript para recrear entrenamiento
│   ├── validation.py  # Script de validacion y clasificacion del dataset
│   └── utils.py       # Funciones recurrentes y sistema de rutas
├── images/
│   ├── confg
│      ├── data.yalm 
│      ├── house_project.v1i.yolov11.zip # Etiquietas generadas desde roboflow
│   ├── train  # Imagenes y etiquetas de entrenamiento
│   ├── test   # Imagenes y etiquetas de evaluación
│   ├── validation   # Imagenes y etiquetas de validación
├── models/
│   ├── train
│   ├── test
│   ├── validation
├── .gitignore           # Archivos y carpetas excluidos del control de versiones
├── README.md            # Documentación del proyecto
└── requirements.txt     # Lista de dependencias del proyecto
```


# Requerimientos

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics)


```bash
pip install ultralytics==8.4.21

pip install supervision==0.27.0.post1

pip install albumentations==2.0.8
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

git clone https://github.com/<usuario>/YOLO-house-identifier.git
cd YOLO-house-identifier

### 2. Instalar dependencias

Instalar las dependencias definidas en el archivo requirements.txt:

pip install -r requirements.txt

as principales librerías utilizadas son:

ultralytics

supervision

albumentations

numpy

### 3. Preparar el dataset

Descargar el dataset desde el repositorio público de Google Drive:

https://drive.google.com/drive/folders/1F0ZShSpEq7DVzTN4xrlTPYH8QZA--fTg

Una vez descargado, ubicar la estructura de carpetas dentro del directorio images/ del proyecto.

El archivo de configuración del dataset se encuentra en:

images/conf/house.project.v1.yolo11.zip

### 4. Entrenar el modelo

Para entrenar el modelo ejecutar:

python src/train_yolo.py

Este script entrena el modelo YOLO utilizando el dataset definido en data.yaml.

Los pesos generados durante el entrenamiento se almacenan en la carpeta:

models/runs_house_model/house_yolo/weights

### 5. Entrenar el modelo

Para realizar detección de casas sobre imágenes nuevas ejecutar:

python src/inference.py

El script cargará el modelo entrenado y generará detecciones sobre las imágenes de prueba, dibujando los bounding boxes correspondientes.

### 6. Entrenar el modelo

Para evaluar el desempeño del modelo y calcular métricas como falsos positivos (FP) y falsos negativos (FN), ejecutar:

python models/runs_house_model/house_yolo

Esto analiza las predicciones del modelo sobre el conjunto de validación y genera métricas de desempeño.

---

# Resultados (métricas) y ejemplos de detección

## Resultados (métricas) y ejemplos de detección

El modelo fue evaluado utilizando el conjunto de validación definido en el dataset.

Las métricas principales utilizadas fueron:

- **Precision**
- **Recall**
- **mAP@0.5**
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

En la siguientes imágen el modelo logra identificar correctamente la fachadas de una casa presente en la escena.

/content/train/images/real_091_San_Andr_s_San_Andr_s_y_Providencia_Colombia_-_panoramio_2__png.rf.8dbe952a3b3d564f61e63201a765edb5.jpg

### Ejemplos de errores de detección

Se identificaron algunos casos donde el modelo presenta errores:

**Falsos positivos (FP)**  
El modelo detecta una casa en objetos visualmente similares, como edificios o estructuras arquitectónicas.

**Falsos negativos (FN)**  
El modelo no detecta casas cuando:

- la fachada está parcialmente oculta
- la iluminación es baja
- la casa aparece muy pequeña en la imagen

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












