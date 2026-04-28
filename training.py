from ultralytics import YOLO
from ultralytics import settings
from config import DEVICE, EPOCHS, PATH_CONFIG_YAML, PROJECT, MODEL, PLATFORM, BEST_MODEL_PATH
from print_metrics import print_metricas

print(f'Ultralytics settings: {settings}')

if __name__ == '__main__':
    #instancia modelo inicial
    yolo_custom = YOLO("yolov8n.pt")

    print(f'YOLO training on platform {PLATFORM} using device {DEVICE}')
    print(f'YAML config path: {PATH_CONFIG_YAML}')
    print(f'Project {PROJECT}, Model {MODEL}, Epochs {EPOCHS}')
    results_treino = yolo_custom.train(
        data=PATH_CONFIG_YAML,
        epochs=EPOCHS,
        imgsz=640,
        batch=8,
        device=DEVICE,
        project=PROJECT,
        name=MODEL,
        exist_ok=True,
        #patience=15,
        plots=True,
        amp=False,
        verbose=False
    )

    #instancia melhor modelo treinado
    yolo_best = YOLO(BEST_MODEL_PATH)

    print_metricas(yolo_best, 'val', PATH_CONFIG_YAML)
    print_metricas(yolo_best, 'test', PATH_CONFIG_YAML)