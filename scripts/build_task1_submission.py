"""Build a reproducible Task 1 handoff ZIP; no raw course images or unrelated tasks."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/task1/kaggle-full-16ay9832'


def build(output):
    output = Path(output).resolve()
    run_dir = ROOT / 'models/task1'
    run = json.loads((run_dir/'run.json').read_text())
    named = [
        'README.md','SUBMISSION.md','pyproject.toml','uv.lock','requirements-inference.txt',
        'task1_demo.py','launch_task1_demo.bat',
        'notebooks/task1-sota.ipynb','notebooks/00_eda_and_preprocessing.ipynb',
        'src/__init__.py','src/preprocessing.py','src/task1_models.py','src/task1_inference.py',
        'scripts/build_task1_submission.py','scripts/validate_task1_delivery.py',
        'docs/REPORT_TASK1.md','docs/INDEPENDENT_EVALUATION_TASK1.md',
        'docs/TASK1_PATCH_NOTES.md','docs/TASK1_DELIVERY_VALIDATION.json',
        'preprocessed_datasets/train_manifest.csv','datasets/test/styles_prediction.csv',
        'artifacts/task1/kaggle-full-16ay9832/task1_ddp_runtime.py',
    ]
    paths = [ROOT/name for name in named]
    paths += [run_dir/'run.json',*[run_dir/name for name in run['files']]]
    paths = sorted(set(paths))
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing required delivery files: {missing}')
    for name,record in run['files'].items():
        if hashlib.sha256((run_dir/name).read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('Original artifact hash mismatch: '+name)
    if output in paths:
        raise ValueError('Output cannot overwrite an input file.')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError(f'{output} already exists. Choose a new --output to preserve previous packages.')
    manifest = {'scope':'Task 1 only; merge Tasks 2-4 before final submission.',
                'run':'task1_full_16ay9832','source_notebook':'notebooks/task1-sota.ipynb',
                'original_deployment_source_notebook_is_historical':True,
                'datasets':'Raw course images are supplied separately in the README layout.',
                'files':{}}
    # Fixed ZIP timestamps and ordering make builds from identical inputs deterministic.
    def write(z,name,data):
        info=zipfile.ZipInfo('task1_handoff/'+name,date_time=(2026,9,10,0,0,0))
        info.compress_type=zipfile.ZIP_DEFLATED
        info.external_attr=0o644 << 16
        z.writestr(info,data,compresslevel=6)
    try:
        with zipfile.ZipFile(output,'x') as archive:
            for p in paths:
                name=p.relative_to(ROOT).as_posix();data=p.read_bytes()
                manifest['files'][name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
                write(archive,name,data)
            write(archive,'PACKAGE_MANIFEST.json',(json.dumps(manifest,indent=2)+'\n').encode())
        with zipfile.ZipFile(output) as archive:
            bad=archive.testzip()
            if bad:raise ValueError('ZIP integrity error: '+bad)
            for name,record in manifest['files'].items():
                if hashlib.sha256(archive.read('task1_handoff/'+name)).hexdigest()!=record['sha256']:
                    raise ValueError('Packaged file mismatch: '+name)
    except Exception:
        # Do not silently delete a partial package; report the failure for inspection.
        raise
    print(f'Built and verified {len(manifest["files"])} files: {output}')
    print(f'Size: {output.stat().st_size/1e6:.1f} MB. Task 1 component, not the complete assignment.')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/task1_handoff_20260910.zip')
    build(parser.parse_args().output)

