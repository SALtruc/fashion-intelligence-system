"""Load the selected Task 1 checkpoint without executing training cells."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import time
import numpy as np
from PIL import Image, ImageOps
import torch
from .task1_models import SmallResNet, PlainCNN

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = ROOT / 'artifacts' / 'task1'


def load_image(path):
    with Image.open(path) as image:
        image.load()
        return ImageOps.exif_transpose(image).convert('RGB')


def standardize_image(image, size):
    image = ImageOps.exif_transpose(image).convert('RGB')
    if image.size == tuple(size):
        return image
    return ImageOps.pad(image, tuple(size), method=Image.Resampling.BILINEAR,
                        color=(255, 255, 255), centering=(0.5, 0.5))


class Predictor:
    def __init__(self, artifacts=DEFAULT_ARTIFACTS, device='cpu'):
        self.artifacts = Path(artifacts)
        self.device = torch.device(device)
        self.deployment = json.loads((self.artifacts / 'deployment.json').read_text())
        selection = json.loads((self.artifacts / 'selection.json').read_text())
        self.arm = self.deployment['arm']
        if self.arm != selection['winner']:
            raise ValueError('Deployment and selection disagree about the selected model.')
        # Original metadata records a basename, and run.json keys it under final/. The
        # folder on Drive names that directory models/, so both are searched rather than
        # asking whoever restores the artifacts to rename a directory first. The hash
        # check below is what actually decides the file is the right one.
        name = self.deployment['artifact']
        if Path(name).name != name:
            raise ValueError('Expected a checkpoint basename in deployment metadata.')
        candidates = [self.artifacts / folder / name for folder in ('final', 'models')]
        candidates.append(self.artifacts / name)
        checkpoint_path = next((p for p in candidates if p.is_file()), None)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f'{name} not found under {self.artifacts}. Looked in: '
                + ', '.join(str(p.parent) for p in candidates))
        run = json.loads((self.artifacts / 'run.json').read_text())
        self.sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if self.sha256 != run['files']['final/' + name]['sha256']:
            raise ValueError('Checkpoint hash differs from the recorded run.')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        for field in ('arm', 'family', 'classes', 'recipe', 'image_target_size',
                      'normalisation_mean', 'normalisation_std'):
            if checkpoint[field] != self.deployment[field]:
                raise ValueError(f'Checkpoint and deployment disagree on {field}.')
        self.classes = checkpoint['classes']
        self.size = checkpoint['image_target_size']
        factory = {'resnet': SmallResNet, 'cnn': PlainCNN}[checkpoint['family']]
        self.model = factory(n_classes=len(self.classes)).to(self.device).eval()
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.mean = torch.tensor(checkpoint['normalisation_mean'], device=self.device).view(1,3,1,1)
        self.std = torch.tensor(checkpoint['normalisation_std'], device=self.device).view(1,3,1,1)
        if not torch.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError('Invalid normalization parameters.')

    @torch.inference_mode()
    def scores(self, images):
        if not images:
            raise ValueError('Provide at least one image.')
        arrays = np.stack([np.asarray(standardize_image(im, self.size)) for im in images])
        x = torch.from_numpy(arrays.copy()).to(self.device).permute(0,3,1,2).float() / 255.0
        logits = self.model((x - self.mean) / self.std)
        if not torch.isfinite(logits).all():
            raise RuntimeError('Model produced non-finite scores.')
        return logits.softmax(1).cpu().numpy()

    def predict(self, image, top_k=5):
        if not 1 <= top_k <= len(self.classes):
            raise ValueError('top_k is outside the label range.')
        start = time.perf_counter()
        scores = self.scores([image])[0]
        indices = np.argsort(-scores, kind='stable')[:top_k]
        return {'articleType': self.classes[int(indices[0])], 'arm': self.arm,
                'suggestions': [{'label': self.classes[int(i)], 'score': float(scores[i])} for i in indices],
                'seconds': time.perf_counter() - start,
                'score_note': 'Softmax scores are not calibrated correctness probabilities.',
                'checkpoint_sha256': self.sha256}


def predict_template(predictor, template_path, image_dir, output_path, batch_size=32):
    template_path, output_path = Path(template_path), Path(output_path)
    if output_path.resolve() == template_path.resolve():
        raise ValueError('Write predictions to a new file, preserving the original template.')
    if batch_size < 1:
        raise ValueError('batch_size must be positive.')
    with template_path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)
    if columns != ['id','gender','articleType','season','usage'] or not rows:
        raise ValueError('Unexpected or empty assignment prediction template.')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate prediction IDs.')
    if any(not re.fullmatch(r'[0-9]+', row['id']) for row in rows):
        raise ValueError('Image IDs must be numeric filenames.')
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        scores = predictor.scores([load_image(Path(image_dir) / (row['id'] + '.jpg')) for row in batch])
        for row, i in zip(batch, scores.argmax(1)):
            row['articleType'] = predictor.classes[int(i)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    parser.add_argument('--threads', type=int, default=4)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--image', type=Path)
    mode.add_argument('--template', type=Path)
    parser.add_argument('--image-dir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    predictor = Predictor(args.artifacts, args.device)
    if args.image:
        print(json.dumps(predictor.predict(load_image(args.image)), indent=2))
    else:
        if not args.image_dir or not args.output:
            parser.error('--template requires --image-dir and --output')
        print(f'Wrote {predict_template(predictor, args.template, args.image_dir, args.output)} predictions to {args.output}')


if __name__ == '__main__':
    main()


