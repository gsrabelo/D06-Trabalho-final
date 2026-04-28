from ultralytics import YOLO
from ultralytics import settings
import platform

projeto = "transfer_v1_ep30"
nome_modelo = "yolo_transfer"
best_model_path = f'./runs/detect/{projeto}/{nome_modelo}/weights/best.pt'

if platform.system() == "Windows":
    print("Running on Windows")
    path_config_yaml = 'config-win.yaml'
    settings.update({'runs_dir': 'C:\\00 IA\\D06 Trabalho final\\runs'})
    settings.update({'datasets_dir': 'C:\\00 IA\\D06 Trabalho final\\dataset'})
    settings.update({'weights_dir': 'C:\\00 IA\\D06 Trabalho final\\weights'})

if platform.system() == "Darwin":
    print("Running on macOS")
    path_config_yaml = 'config.yaml'

print(f'Ultralytics settings: {settings}')

def print_metricas(model, particao):
    if particao not in ('val', 'test'):
        print('Partição inválida. Use "val" ou "test".')
        return
    
    metricas = model.val(data=path_config_yaml, verbose=False, split=particao)
    print('\n\n--------------------------------')
    print('Métricas partição', particao)
    print('------------------------------------')
    print(f'    mAP@0.50                  : {metricas.box.map50:0.3f}')
    print(f'    mAP@0.50:0.95             : {metricas.box.map:0.3f}')
    print(f'    Precisão                  : {metricas.box.mp:0.3f}')
    print(f'    Recall médio              : {metricas.box.mr:0.3f}')
    print(60*'-')
    for i, cls_idx in enumerate(metricas.box.ap_class_index):
        nome = model.names[cls_idx]
        ap = metricas.box.ap50[i]
        print(f'  {int(cls_idx)} {nome:<15} AP@50 = {ap:.4f}')

if __name__ == '__main__':
    #instancia modelo inicial
    yolo_custom = YOLO("yolov8n.pt")
    yolo_best = YOLO(best_model_path)

    print_metricas(yolo_best, 'val')
    print_metricas(yolo_best, 'test')