from ultralytics import YOLO
from src.print_metrics import print_metricas
from src.config import PROJECT, MODEL, BEST_MODEL_PATH, PATH_CONFIG_YAML

if __name__ == '__main__':
    #instancia modelo inicial
    yolo_best = YOLO(BEST_MODEL_PATH)

    print_metricas(yolo_best, 'val', PATH_CONFIG_YAML)
    print_metricas(yolo_best, 'test', PATH_CONFIG_YAML)