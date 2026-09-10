"""Export the official dual-environment notebook for Kaggle; refresh the dataset bundle."""
import ast
from pathlib import Path
import subprocess
import sys
import nbformat
ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'kaggle'
nb = nbformat.read(ROOT/'notebooks/01_task1_article_type_classification.ipynb', 4)
assert nb.cells[1].source.startswith('KAGGLE = False')
nb.cells[1].source = nb.cells[1].source.replace('KAGGLE = False', 'KAGGLE = True', 1)
nb.metadata['kaggle'] = dict(isGpuEnabled=True, isInternetEnabled=True)
for cell in nb.cells:
    if cell.cell_type == 'code':
        cell.outputs = []
        cell.execution_count = None
        compile(cell.source, 'cell', 'exec')
nbformat.validate(nb)
nbformat.write(nb, DEST/'task1_kaggle_full_run.ipynb')
launcher = next(c.source for c in nb.cells if c.cell_type == 'code' and c.source.startswith('RUNTIME_SOURCE ='))
runtime = ast.literal_eval(ast.parse(launcher).body[0].value)
compile(runtime, 'runtime', 'exec')
(DEST/'task1_ddp_runtime.py').write_text(runtime, encoding='utf-8')
(DEST/'README.md').write_text(nb.cells[0].source, encoding='utf-8')
print('Exported official notebook with KAGGLE=True')
if '--notebook-only' not in sys.argv:
    subprocess.run([sys.executable, str(DEST/'refresh_bundle.py')], check=True, cwd=ROOT)
