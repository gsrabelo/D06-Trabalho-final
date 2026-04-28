from ultralytics import YOLO

def print_metricas(model, particao, path_config_yaml):
    if particao not in ('val', 'test'):
        print('Partição inválida. Use "val" ou "test".')
        return
    
    metricas = model.val(data=path_config_yaml, verbose=False, split=particao)
    print('\n\n------------------------------------')
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