import platform
from ultralytics import settings

# Default parameters
DEVICE = "cpu"
PLATFORM = platform.system() 
PATH_CONFIG_YAML = 'config-win.yaml'
PROJECT = "transfer_v1_ep30"
MODEL = "yolo_transfer"
EPOCHS = 3
BEST_MODEL_PATH = f'./runs/detect/{PROJECT}/{MODEL}/weights/best.pt'
MY_MODEL_PATH = './models/best.pt'
YOLO_MODEL_PATH = './models/yolov8n.pt'

def set_device():
    if PLATFORM == "Windows":
        return "cpu"
    elif PLATFORM == "Darwin":
        return "mps"
    else:
        return "cpu"

def set_path_config_yaml():
    if PLATFORM == "Windows":
        return 'config-win.yaml'
    elif PLATFORM == "Darwin":
        return 'config-mac.yaml'
    else:
        return 'config-win.yaml'

def set_ultralytics_settings():
    if PLATFORM == "Windows":
        settings.update({'runs_dir': 'C:\\00 IA\\teste\\D06-Trabalho-final\\runs'})
        settings.update({'datasets_dir': 'C:\\00 IA\\teste\\D06-Trabalho-final\\data\\dataset'})
        settings.update({'weights_dir': 'C:\\00 IA\\teste\\D06-Trabalho-final\\weights'})

def get_config():
    config ={
        'platform': PLATFORM,
        'device': set_device(),
        'path_config_yaml': set_path_config_yaml(),
        'project': PROJECT,
        'model': MODEL,
        'epochs': EPOCHS,
        'best_model_path': BEST_MODEL_PATH,
        'my_model_path': MY_MODEL_PATH,
        'yolo_model_path': YOLO_MODEL_PATH
    }
    set_ultralytics_settings()
    return config

if __name__ == '__main__':
    print(f'Ultralytics settings: {settings}')
