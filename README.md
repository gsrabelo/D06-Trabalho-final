# Projeto Final - Visão Computacional e Modelos Generativos

Aluno: Guilherme Silveira Rabelo  
Professor: João José de Macedo Neto

Este modelo detecta duas classes de armas de fogo em imagens: pistola e fuzil.

O modelo foi treinado usando como base o modelo pré-treinado yolov8n.pt e um dataset de 101 imagens para treinamento e 15 para validação.

## Preparando o ambiente

### 1. Baixando o código

```
git clone https://github.com/gsrabelo/D06-Trabalho-final.git
cd D06-Trabalho-final
```

### 2. Preparando o ambiente virtual usando o gerenciador de pacotes UV

#### Windows
```
uv venv --python 3.12
.venv\Scripts\activate
```

#### Linux/Mac
```
uv venv --python 3.12
source .venv/bin/activate
```

### 3. Instalando as dependências
```
uv pip install -r requirements.txt
```

## Rodando o código de treinamento e validação/teste

### 1. Conferindo as configurações globais (src/config.py)

Exemplo: 

```
DEVICE = "cpu"
PLATFORM = platform.system() 
PATH_CONFIG_YAML = 'config-win.yaml'
PROJECT = "transfer_v1_ep30"
MODEL = "yolo_transfer"
EPOCHS = 50
BEST_MODEL_PATH = f'./runs/detect/{PROJECT}/{MODEL}/weights/best.pt'
MY_MODEL_PATH = './models/best.pt'
YOLO_MODEL_PATH = './models/yolov8n.pt'
RUNS_DIR_WIN = 'C:\\00 IA\\D06-Trabalho-final\\runs'
DATASETS_DIR_WIN = 'C:\\00 IA\\D06-Trabalho-final\\data\\dataset'
WEIGHTS_DIR_WIN = 'C:\\00 IA\\D06-Trabalho-final\\weights'
```

Atentar para o OS e tipo de dispositivo (cpu, gpu ou mps).

### 2. Conferindo o arquivo de configuração do YOLO

Dois arquivos possíveis: um para WINDOWS (config-win.yaml) e outro para MAC (config-macos.yaml)

Customizar o arquivo de configuração conforme o OS e a pasta do seu projeto.

Exemplo:

```
path: [Sua pasta de trabalho]/D06-Trabalho-final/data/dataset
train: train/images
val: val/images
test: test/images

# nome das classes
names:
 0: Pistola
 1: Fuzil
```

Atenção: não inverter a ordem das classes

### 3. Caso queira treinar novamente o modelo

Executar o comando a partir da pasta raiz do projeto:

```
python -m src.train
```

CUIDADO: o programa irá sobreescrever os modelos treinados anteriormente.

### 4. Validando e testando o modelo

Executar o comando a partir da pasta raiz do projeto:

```
python -m src.validate
```

Conferir métricas nas pastas `runs/detect/val`

## Testando o modelo previamente treinado com notebook Jupyter

Abrir o arquivo `notebooks/teste_modelo_custom.ipynb` e executar as células de código, observando as informações de saída.

ATENÇÃO: A última célula depende de um arquivo de vídeo que não se encontra no github. Pegar arquivo manualmente e colocar na seguinte pasta e com o seguinte nome:  

 `data/video/video_teste.mov` 


