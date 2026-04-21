import os
print(os.getcwd())

from pathlib import Path
print(Path.cwd())

import sys
print(sys.executable)

import sys
for path in sys.path:
    print(path)

import os
print(os.environ.get('PYTHONPATH'))

from ultralytics import settings

# Atualizar o diretório de runs
#settings.update({'runs_dir': 'C:\\00 IA\\D06 Trabalho final\\runs'})
#settings.update({'datasets_dir': 'C:\\00 IA\\D06 Trabalho final\\dataset'})
print(settings)