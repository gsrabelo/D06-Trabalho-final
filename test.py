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