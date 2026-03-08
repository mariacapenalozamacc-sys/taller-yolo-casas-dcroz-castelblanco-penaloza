
# Proyecto YOLO - Identificación de Casas con Modelo Basado en YOLO


# Descripción general del proyecto



# Estructura del repositorio


```text
YOLO-house-identifier/
│
├── examples/            
│
├── src/
│   ├── inferencia.py    # Script principal para ejecutar el sistema
│   ├── train_yolo.py 
│   ├── validation.py
│   └── utils.py 
├── images/
│   ├── train
│   ├── test
│   ├── validation
├── models/
│   ├── train
│   ├── test
│   ├── validation
├── .gitignore           # Archivos y carpetas excluidos del control de versiones
├── README.md            # Documentación del proyecto
└── requirements.txt     # Lista de dependencias del proyecto
```


# Requerimientos

```bash
pip install ultralytics==8.4.21

pip install supervision==0.27.0.post1

pip install albumentations==2.0.8
```

# Construcción de la Herramienta

## Datos de entrenamiento

Datos en este link: https://drive.google.com/drive/folders/1F0ZShSpEq7DVzTN4xrlTPYH8QZA--fTg?usp=drive_link


## Arquitectura del modelo


