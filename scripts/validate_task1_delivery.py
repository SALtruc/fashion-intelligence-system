"""Artifact-backed checks for the Task 1 delivery; does not train models."""
import csv
import hashlib
import json
import pathlib
import sys
import tempfile
import time
import ast
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.task1_inference import Predictor, DEFAULT_ARTIFACTS, load_image, predict_template, standardize_image
from PIL import Image


def main():
    torch.set_num_threads(4)
    artifacts = DEFAULT_ARTIFACTS
    manifest = json.loads((artifacts/'run.json').read_text())
    for name,record in manifest['files'].items():
        assert hashlib.sha256((artifacts/name).read_bytes()).hexdigest() == record['sha256'],name
    print(f"Verified {len(manifest['files'])} original artifact hashes.",flush=True)
    predictor = Predictor()
    with (artifacts/'predictions/reporting_all_models.csv').open() as f:
        reporting = list(csv.DictReader(f))
    # First observation of every observed class, plus a contiguous batch.
    sampled = {}
    for row in reporting:
        sampled.setdefault(row['articleType'],row)
    samples = list({row['id']:row for row in [*sampled.values(),*reporting[:32]]}.values())
    predictions=[]
    start=time.perf_counter()
    for offset in range(0,len(samples),16):
        batch=samples[offset:offset+16]
        scores=predictor.scores([load_image(ROOT/'datasets/train/images_train'/f"{row['id']}.jpg") for row in batch])
        assert np.isfinite(scores).all() and np.allclose(scores.sum(1),1,atol=1e-5)
        predictions.extend(scores.argmax(1).tolist())
    mismatches=[{'id':row['id'],'saved':int(row[predictor.arm]),'cpu':pred} for row,pred in zip(samples,predictions) if pred!=int(row[predictor.arm])]
    print(f'CPU FP32 replay: {len(samples)-len(mismatches)}/{len(samples)} saved top-1 labels matched in {time.perf_counter()-start:.2f}s.',flush=True)
    if mismatches:
        print('Numerical/runtime differences:',mismatches,flush=True)
    # Normalization/image transform must match the executed implementation, not a mirrored test.
    runtime=(ROOT/'artifacts/task1/kaggle-full-16ay9832/task1_ddp_runtime.py').read_text(encoding='utf-8')
    tree=ast.parse(runtime)
    original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='standardize_image')
    from PIL import ImageOps
    scope={'Image':Image,'ImageOps':ImageOps,'IMAGE_TARGET_SIZE':(60,80),'IMAGE_PAD_RGB':(255,255,255)}
    exec(compile(ast.Module(body=[original],type_ignores=[]),'original_transform','exec'),scope)
    rng=np.random.default_rng(42)
    for shape in [(80,60,3),(110,70,3),(50,180,3)]:
        im=Image.fromarray(rng.integers(0,256,size=shape,dtype=np.uint8))
        assert np.array_equal(np.asarray(standardize_image(im,(60,80))),np.asarray(scope['standardize_image'](im)))
    with tempfile.TemporaryDirectory(prefix='task1_delivery_') as folder:
        temp=pathlib.Path(folder)
        source=ROOT/'datasets/test/styles_prediction.csv'
        with source.open() as f:
            reader=csv.DictReader(f);columns=reader.fieldnames;rows=list(reader)[:5]
        rows[0]['season']='keep-existing-value'
        template=temp/'template.csv'
        with template.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
        output=temp/'predicted.csv'
        assert predict_template(predictor,template,ROOT/'datasets/test/images_test',output)==5
        with output.open() as f:actual=list(csv.DictReader(f))
        assert [r['id'] for r in actual]==[r['id'] for r in rows]
        assert actual[0]['season']=='keep-existing-value'
        assert all(row['articleType'] in predictor.classes for row in actual)
        try:
            predict_template(predictor,template,ROOT/'datasets/test/images_test',template)
            raise AssertionError('Template overwrite was allowed.')
        except ValueError:pass
        # Exercise real GUI state transitions and async prediction without leaving a window open.
        import tkinter as tk
        from task1_demo import CatalogueApp
        root=tk.Tk();root.withdraw()
        app=CatalogueApp(root);app.predictor=predictor
        image_path=ROOT/'datasets/test/images_test'/f"{rows[0]['id']}.jpg"
        app.classify_path(image_path)
        deadline=time.monotonic()+40
        while app.future is not None and time.monotonic()<deadline:
            root.update();time.sleep(.03)
        assert app.result is not None,'GUI prediction failed or timed out'
        assert len(app.ranking.get_children())==5
        app.export_review(temp/'review.csv')
        with (temp/'review.csv').open() as f:review=list(csv.DictReader(f))
        assert review[0]['reviewed_articleType']==app.result['articleType']
        bad=temp/'broken.jpg';bad.write_bytes(b'not an image')
        app.classify_path(bad)
        deadline=time.monotonic()+10
        while app.future is not None and time.monotonic()<deadline:
            root.update();time.sleep(.03)
        assert app.result is None and str(app.save['state'])=='disabled'
        app.close()
    result={'artifact_hashes_verified':len(manifest['files']),'sample_rows':len(samples),
            'sample_classes':len(sampled),'cpu_top1_mismatches':mismatches,
            'transform_matches_original':True,'template_preservation':True,
            'gui_prediction_review_export_and_invalid_image':True,'torch':torch.__version__,
            'note':'CPU FP32 replay on a representative sample; not a full retraining or external evaluation.'}
    (ROOT/'docs/TASK1_DELIVERY_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    assert not mismatches,'Investigate replay differences before delivery.'
    print('Transform, prediction-template, GUI prediction/export/error checks passed.',flush=True)


if __name__=='__main__':main()

