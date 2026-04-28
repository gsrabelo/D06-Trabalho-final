from ultralytics import YOLO
from ultralytics import settings
from src.config import get_config
from src.print_metrics import print_metricas

config = get_config()
print(f'Ultralytics settings: {settings}')

if __name__ == '__main__':
    #instancia modelo inicial
    yolo_custom = YOLO(config['yolo_model_path'])

    print(f'YOLO training on platform {config["platform"]} using device {config["device"]}')
    print(f'YAML config path: {config["path_config_yaml"]}')
    print(f'Project {config["project"]}, Model {config["model"]}, Epochs {config["epochs"]}')
    results_treino = yolo_custom.train(
        data=config['path_config_yaml'],
        epochs=config['epochs'],
        imgsz=640,
        batch=8,
        device=config['device'],
        project=config['project'],
        name=config['model'],
        exist_ok=True,
        #patience=15,
        plots=True,
        amp=False,
        verbose=False
    )

    #instancia melhor modelo treinado
    yolo_best = YOLO(config['best_model_path'])

    print_metricas(yolo_best, 'val', config['path_config_yaml'])
    print_metricas(yolo_best, 'test', config['path_config_yaml'])