# Projeto Final - Visão Computacional e Modelos Generativos

Aluno: Guilherme Silveira Rabelo  
Professor: João José de Macedo Neto

Este modelo detecta duas classes de armas de fogo em imagens: pistola e fuzil.

O modelo foi treinado usando como base o modelo pré-treinado yolov8n.pt e um dataset de 101 imagens para treinamento e 15 para validação.

## Preparando o ambiente

### 1. Baixando o código

```bash
git clone https://github.com/gsrabelo/D06-Trabalho-final.git
cd D06-Trabalho-final
```

### 2. Preparando o ambiente usando o gerenciador de pacotes UV

#### Windows
```bash
uv venv --python 3.12
.venv\Scripts\activate
```

#### Linux/Mac
```bash
uv venv --python 3.12
source .venv/bin/activate
```

#### Instalando as dependências se for treinar novamente o modelo
```bash
uv pip install -r requirements-training.txt
```

#### Instalando as dependências se for apenas testar o modelo com o notebook jupyter
```bash
uv pip install -r requirements-testing.txt
```

### 3. Rodando o código

#### Conferindo as configurações globais (config.py)
```bash
DEVICE = "cpu"
PLATFORM = platform.system() 
PATH_CONFIG_YAML = 'config-win.yaml'
PROJECT = "transfer_v1_ep30"
MODEL = "yolo_transfer"
EPOCHS = 3
BEST_MODEL_PATH = f'./runs/detect/{PROJECT}/{MODEL}/weights/best.pt'
```

Atentar para o OS e tipo de dispositivo (cpu, gpu ou mps).

#### Conferindo o arquivo de configuração do YOLO

Dois arquivos possíveis: um para WINDOWS (config-win.yaml) e outro para MAC (config-macos.yaml)

Customizar o arquivo de configuração conforme o OS e a pasta do seu projeto.

