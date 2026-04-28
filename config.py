import platform
from ultralytics import settings

DEVICE = "cpu"
PLATFORM = platform.system() 
PATH_CONFIG_YAML = 'config-win.yaml'
PROJECT = "transfer_v1_ep30"
MODEL = "yolo_transfer"
EPOCHS = 50
BEST_MODEL_PATH = f'./runs/detect/{PROJECT}/{MODEL}/weights/best.pt'

if PLATFORM == "Windows":
    DEVICE = "cpu"
    PATH_CONFIG_YAML = 'config-win.yaml'
    settings.update({'runs_dir': 'C:\\00 IA\\D06 Trabalho final\\runs'})
    settings.update({'datasets_dir': 'C:\\00 IA\\D06 Trabalho final\\dataset'})
    settings.update({'weights_dir': 'C:\\00 IA\\D06 Trabalho final\\weights'})

if PLATFORM == "Darwin":
    DEVICE = "mps"
    PATH_CONFIG_YAML = 'config-mac.yaml'

if __name__ == '__main__':
    print(f'Ultralytics settings: {settings}')
