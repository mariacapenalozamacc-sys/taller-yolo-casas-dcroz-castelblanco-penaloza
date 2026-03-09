
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

## Entrenamiento 


