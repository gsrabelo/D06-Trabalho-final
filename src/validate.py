from ultralytics import YOLO
from ultralytics import settings
from src.print_metrics import print_metricas
from src.config import get_config

config = get_config()

# RUN IF PATH ERROR IN ULTRALYTICS
#set_ultralytics_settings()

print(f'Ultralytics settings: {settings}')

if __name__ == '__main__':
    #instancia modelo inicial
    yolo_best = YOLO(config['best_model_path'])

    print_metricas(yolo_best, 'val', config['path_config_yaml'])
    print_metricas(yolo_best, 'test', config['path_config_yaml'])