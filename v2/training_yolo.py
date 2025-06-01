from ultralytics import YOLO
import os

# Configuración de directorios
DATA_DIR = "data2"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

# Cargar el modelo YOLO pose
#model = YOLO("yolov11x-pose.pt")  # cargar modelo pre-entrenado
model = YOLO("yolov8n-pose.pt")  # cargar modelo pre-entrenado

# Configurar y ejecutar el entrenamiento
results = model.train(
    data="data.yaml",          # archivo de configuración del dataset
    #epochs=100,                   # número de épocas
    epochs=2,                   # número de épocas
    imgsz=640,                    # tamaño de imagen
    batch=0.9,                    # tamaño del batch
    device="0",                   # GPU a usar (0 para primera GPU)
    workers=8,                    # número de workers para data loading
    patience=50,                  # early stopping patience
    save=True,                    # guardar mejores pesos
    save_period=10,              # guardar cada 10 épocas
    cache=True,                  # cachear imágenes
    exist_ok=False,              # no sobrescribir directorio de salida
    pretrained=True,             # usar pesos pre-entrenados
    optimizer="auto",            # optimizador automático
    verbose=True,                # mostrar información detallada
    cos_lr=True,                #  usar cosine learning rate
    amp=True,                   # usar mixed precision training
    val=True,                   # validar durante el entrenamiento
)

# Validar el modelo
metrics = model.val()
print("\nMétricas de validación:")
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
print(f"Precisión: {metrics.box.precision:.3f}")
print(f"Recall: {metrics.box.recall:.3f}")
print(f"F1-score: {metrics.box.f1:.3f}")
