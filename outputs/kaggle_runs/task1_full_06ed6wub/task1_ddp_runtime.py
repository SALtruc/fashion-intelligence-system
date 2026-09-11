import os, sys
from pathlib import Path
RUN_ROOT = Path(os.environ['TASK1_RUN_ROOT'])
WORK_ROOT = Path(os.environ['TASK1_WORK_ROOT'])
os.chdir(RUN_ROOT)
rank = int(os.environ.get('LOCAL_RANK', 0))
size = int(os.environ.get('WORLD_SIZE', 1))
ALL_CPU_IDS = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count() or 1))
if hasattr(os, 'sched_getaffinity'):
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, cpus[rank::size] or [cpus[rank % len(cpus)]])
if rank:
    sys.stdout = open(os.devnull, 'w')
import matplotlib
matplotlib.use('Agg')

import os
import platform

# sched_getaffinity is the right question to ask rather than cpu_count: inside a container
# or under taskset it reports the cores this process may actually use. It does not exist on
# Windows or macOS, hence the fallback.
try:
    VISIBLE_CORES = len(os.sched_getaffinity(0))
except AttributeError:
    VISIBLE_CORES = os.cpu_count() or 1

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
                  "LOKY_MAX_CPU_COUNT"):
    os.environ[_variable] = str(VISIBLE_CORES)

HOST_OS = platform.system()
print(f"Host: {HOST_OS} {platform.machine()} | Python {platform.python_version()}")
print(f"CPU : {VISIBLE_CORES} logical cores, all of them")

import gc
import hashlib
import json
import math
import random
import shutil
import time
import warnings
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F

import joblib
from skimage.feature import hog
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Only the warning families that are genuinely noise here are silenced. A blanket
# UserWarning filter would also hide sklearn's "y_pred contains labels not in y_true",
# which is precisely the signal the tail-aware metrics in Section 5 exist to reason about.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.*")
warnings.filterwarnings("ignore", message=".*Palette images with Transparency.*")

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 160)
sns.set_theme(style="whitegrid", context="notebook")

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8b6fc0", "#e87ba4"]
MUTED = "#6b7280"

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from datetime import timedelta
RANK = int(os.environ.get('RANK', 0))
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
DDP_ACTIVE = WORLD_SIZE > 1
if DDP_ACTIVE:
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group('nccl', device_id=torch.device('cuda', LOCAL_RANK), timeout=timedelta(hours=2))

def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model

def shared_cpu(function):
    # Each collective call has one shared artifact; ranks follow the same control flow.
    counter = 0
    def call(*args, **kwargs):
        nonlocal counter
        if not DDP_ACTIVE:
            set_phase('cpu:' + function.__name__)
            with full_cpu_phase():
                return function(*args, **kwargs)
        cache = RUN_ROOT / '_ddp_cache'
        cache.mkdir(exist_ok=True)
        target = cache / f'{function.__name__}_{counter:04d}.joblib'
        counter += 1
        if RANK == 0:
            set_phase('cpu:' + function.__name__)
            with full_cpu_phase():
                joblib.dump(function(*args, **kwargs), target)
        dist.barrier()
        return joblib.load(target, mmap_mode='r')
    return call

"""Embedded into the Kaggle notebook; no external import is needed on Kaggle."""
import contextlib
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time


def cpu_affinities():
    if not hasattr(os, 'sched_getaffinity'):
        return {}
    tasks = Path('/proc/self/task')
    ids = [int(p.name) for p in tasks.iterdir()] if tasks.is_dir() else [0]
    result = {}
    for tid in ids:
        try:
            result[tid] = set(os.sched_getaffinity(tid))
        except ProcessLookupError:
            pass
    return result


def assign_affinities(mapping):
    for tid, cpus in mapping.items():
        try:
            os.sched_setaffinity(tid, cpus)
        except ProcessLookupError:
            pass


@contextlib.contextmanager
def full_cpu_phase():
    """Expand rank-zero CPU work, with one native thread per HOG process."""
    global VISIBLE_CORES
    from threadpoolctl import threadpool_limits
    before = cpu_affinities()
    old_cores, old_threads = VISIBLE_CORES, torch.get_num_threads()
    all_cpus = globals().get('ALL_CPU_IDS', list(next(iter(before.values()), range(old_cores))))
    try:
        assign_affinities({tid: set(all_cpus) for tid in before})
        VISIBLE_CORES = len(all_cpus)
        torch.set_num_threads(1)
        with threadpool_limits(limits=1), joblib.parallel_config(backend='loky', inner_max_num_threads=1):
            yield
    finally:
        VISIBLE_CORES = old_cores
        torch.set_num_threads(old_threads)
        # Include native threads created during the phase in the restored rank allocation.
        rank_cpus = next(iter(before.values()), set(all_cpus))
        assign_affinities({tid: before.get(tid, rank_cpus) for tid in cpu_affinities()})


def restore_full_cpu():
    global VISIBLE_CORES, _restored_thread_limits
    from threadpoolctl import threadpool_limits
    all_cpus = globals().get('ALL_CPU_IDS', list(range(VISIBLE_CORES)))
    assign_affinities({tid: set(all_cpus) for tid in cpu_affinities()})
    VISIBLE_CORES = len(all_cpus)
    torch.set_num_threads(VISIBLE_CORES)
    _restored_thread_limits = threadpool_limits(limits=VISIBLE_CORES)
    print(f'CPU allocation restored: {VISIBLE_CORES} available cores', flush=True)


def writable_array(array):
    result = np.ascontiguousarray(array)
    return result if result.flags.writeable else result.copy()


def set_phase(name):
    if globals().get('RANK', 0) != 0:
        return
    try:
        phase_path = RUN_ROOT / 'phase.json'
        now = time.monotonic()
        previous = json.loads(phase_path.read_text()) if phase_path.exists() else None
        if previous:
            with (RUN_ROOT / 'phase_durations.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps(dict(phase=previous['name'], seconds=now-previous['started']))+'\n')
        temporary = RUN_ROOT / 'phase.tmp'
        temporary.write_text(json.dumps(dict(name=name, started=now)), encoding='utf-8')
        temporary.replace(phase_path)
    except (OSError, ValueError):
        pass  # Telemetry must never stop training.


def distributed_evaluate(model, stream, criterion):
    """Evaluate each row once; return complete, ordered logits on every rank."""
    assert not stream.augment and not stream.shuffle and not stream.drop_last
    assert len(stream.labels) > 0
    model = unwrap_model(model)
    model.eval()
    # DDP normally broadcasts buffers at forward entry. Do it once, then bypass
    # the wrapper so different shard lengths require no per-forward collectives.
    for buffer in model.buffers():
        dist.broadcast(buffer, src=0)
    loss_sum = torch.zeros((), device=DEVICE, dtype=torch.float64)
    finite = torch.ones((), device=DEVICE, dtype=torch.int32)
    n_seen, chunks = 0, []
    stream.shard_eval = True
    try:
        with torch.no_grad():
            for images, targets in stream:
                with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=AMP_ENABLED):
                    logits = model(images)
                    loss = criterion(logits, targets)
                finite.mul_(torch.isfinite(loss).to(torch.int32))
                loss_sum += loss.detach().double() * len(targets)
                n_seen += len(targets)
                chunks.append(logits.float())
    finally:
        stream.shard_eval = False
    # Even a rank with zero rows participates in the same collectives.
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite:
        raise FloatingPointError('Non-finite validation loss; this run must not be reported.')
    totals = torch.stack([loss_sum, loss_sum.new_tensor(n_seen)])
    dist.all_reduce(totals)
    assert int(totals[1].item()) == len(stream.labels), 'Validation row count changed'
    length = torch.tensor([n_seen], dtype=torch.long, device=DEVICE)
    lengths = [torch.empty_like(length) for _ in range(WORLD_SIZE)]
    dist.all_gather(lengths, length)
    counts = [int(x.item()) for x in lengths]
    padded = torch.zeros((max(counts), N_CLASSES), device=DEVICE, dtype=torch.float32)
    if chunks:
        padded[:n_seen] = torch.cat(chunks)
    gathered = [torch.empty_like(padded) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered, padded)
    ordered = torch.empty((len(stream.labels), N_CLASSES), device=DEVICE, dtype=torch.float32)
    for rank, values in enumerate(gathered):
        ordered[rank::WORLD_SIZE] = values[:counts[rank]]
    # Only communication is padded. Padding never enters the loss or metrics.
    return (totals[0]/totals[1]).item(), ordered.cpu().numpy()


class GpuTelemetry:
    fields = ['timestamp_utc', 'elapsed_seconds', 'phase', 'gpu_index', 'gpu_uuid',
              'utilization_pct', 'power_w', 'memory_used_mib', 'memory_total_mib', 'temperature_c']

    def __init__(self, root, interval=2.0):
        self.root, self.interval = Path(root), interval
        self.stop_event = threading.Event()
        self.thread = None
        self.warnings, self.samples = [], []
        self.started = time.monotonic()

    def start(self):
        try:
            self.executable = shutil.which('nvidia-smi')
            if not self.executable:
                self.warnings.append('nvidia-smi unavailable; GPU telemetry disabled')
                return
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
        except Exception as error:
            self.warnings.append(str(error))

    def _loop(self):
        try:
            with (self.root/'gpu_telemetry.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fields)
                writer.writeheader()
                while not self.stop_event.is_set():
                    try:
                        result = subprocess.run([self.executable,
                            '--query-gpu=index,uuid,utilization.gpu,power.draw,memory.used,memory.total,temperature.gpu',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
                        if result.returncode:
                            raise RuntimeError(result.stderr.strip() or 'nvidia-smi query failed')
                        try:
                            phase = json.loads((self.root/'phase.json').read_text())['name']
                        except (OSError, ValueError, KeyError):
                            phase = 'startup'
                        for line in csv.reader(result.stdout.splitlines()):
                            if len(line) != 7:
                                continue
                            values = [v.strip() for v in line]
                            row = dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                       elapsed_seconds=time.monotonic()-self.started, phase=phase,
                                       gpu_index=values[0], gpu_uuid=values[1])
                            for key, value in zip(self.fields[5:], values[2:]):
                                try:
                                    row[key] = float(value)
                                except ValueError:
                                    row[key] = None
                            self.samples.append(row)
                            writer.writerow(row)
                        handle.flush()
                    except Exception as error:
                        if str(error) not in self.warnings:
                            self.warnings.append(str(error))
                    self.stop_event.wait(self.interval)
        except Exception as error:
            self.warnings.append(str(error))

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=7)
        try:
            groups = {}
            for row in self.samples:
                groups.setdefault((row['phase'], row['gpu_index']), []).append(row)
            summary = []
            for (phase, gpu), rows in groups.items():
                entry = dict(phase=phase, gpu_index=gpu, samples=len(rows))
                for key in self.fields[5:]:
                    values = [r[key] for r in rows if r[key] is not None]
                    entry['sample_mean_'+key] = sum(values)/len(values) if values else None
                    if key == 'memory_used_mib':
                        entry['peak_memory_used_mib'] = max(values) if values else None
                summary.append(entry)
            (self.root/'gpu_telemetry_summary.json').write_text(json.dumps(dict(
                interval_seconds=self.interval, elapsed_seconds=time.monotonic()-self.started,
                warnings=self.warnings, phase_gpu_summaries=summary), indent=2), encoding='utf-8')
        except Exception as error:
            print('Telemetry summary unavailable:', error)


def package_run(root, work, quick_run):
    """Package after telemetry and the child log have closed."""
    import hashlib
    root, work = Path(root), Path(work)
    output = root/'models/task1'
    telemetry_dir = output/'telemetry'
    telemetry_dir.mkdir(exist_ok=True)
    for name in ('gpu_telemetry.csv', 'gpu_telemetry_summary.json', 'phase_durations.jsonl', 'task1_ddp.log'):
        if (root/name).is_file():
            shutil.copy2(root/name, telemetry_dir/name)
    manifest_path = output/'run.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['files'] = {p.relative_to(output).as_posix(): dict(
            bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p in output.rglob('*') if p.is_file() and p != manifest_path}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    archive = shutil.make_archive(str(work/(root.name+('_QUICK_results' if quick_run else '_results'))),
                                  'zip', root_dir=output.parent, base_dir=output.name)
    print('Results ZIP:', archive, flush=True)
    return archive


# ---------------------------------------------------------------------------------------
# Every setting reused across the notebook is defined once here, so a change is made in one
# place and the whole protocol reads in one screen.
# ---------------------------------------------------------------------------------------
TARGET = "articleType"
RANDOM_STATE = 42

QUICK_RUN = os.environ.get('TASK1_QUICK_RUN', '0') == '1'        # True: a few rows per class and 1-epoch budgets. Structural check only.
USE_AMP = os.environ.get('TASK1_USE_AMP', '1') == '1'  # Set False to diagnose mixed-precision numerical issues.
ALLOW_CPU = False        # see Section 1.4; ResNet training on CPU presents as a hang

# --- The deterministic image transform, from Section 3.1 of notebook 00 ----------------
IMAGE_TARGET_SIZE = (60, 80)      # width, height: the catalogue's modal image size
IMAGE_PAD_RGB = (255, 255, 255)   # catalogue background, used when padding

# --- Splits, Section 2.1 ---------------------------------------------------------------
REPORTING_SHARE = 0.20   # held out of everything, scored once in Section 5
TUNING_SHARE = 0.20      # held out of fitting; every search and selection decision uses it

# --- Optimisation, Section 3.4 ---------------------------------------------------------
BATCH_SIZE = 128
SEARCH_EPOCHS = 24       # reduced budget, shared by every arm of both neural grids
CONFIRM_EPOCHS = 40      # full budget, for the winning arm of each neural family
PATIENCE = 8             # early stopping on tuning macro-F1
WARMUP_EPOCHS = 3
LABEL_SMOOTHING = 0.05   # justified by the documented label noise, notebook 00 Section 3.2.2

# --- Augmentation, Section 2.4 ---------------------------------------------------------
AUG_FLIP_PROBABILITY = 0.5
AUG_ROTATION_DEGREES = 10.0
AUG_TRANSLATE_FRACTION = 0.08
AUG_JITTER_STRENGTH = 0.2

# --- Class imbalance, Section 3.4 ------------------------------------------------------
# Two regimes, both trained and both scored, so Section 5 shows the effect of the choice
# instead of asserting it. They are alternatives, not additions: each applies a 1/count
# correction exactly once, and stacking them would correct twice for an effective
# 1/count**2, which starves the head without helping the tail.
#   "reweight"  uniform sampling, inverse-frequency loss weights   (the original recipe)
#   "resample"  class-balanced sampling, unweighted loss
# The difference that matters is not the strength of the correction, which is the same, but
# what the rare rows are seen as: under "reweight" a one-image class is one fixed image
# scaled by 195, under "resample" it is drawn about as often as Tshirts and so arrives
# under a different augmentation draw each time.
IMBALANCE_REGIMES = ("reweight", "resample")
RESAMPLE_POWER = 1.0     # draw weight is 1/count ** this; 0 is uniform, 1 is fully balanced

# --- Per-class decision offsets, Section 4.3 -------------------------------------------
# The SVM decides by argmax over decision_function, so a per-class bias that grows with
# rarity trades head precision for tail recall, which is the trade macro-F1 rewards. One
# scalar is fitted rather than 124 offsets: 14 of the 32 rare classes have no tuning row at
# all, and a free offset per class would be fitted on a single image wherever it could be
# fitted at all. Costs no training, so it is scored as its own arm.
SVM_OFFSET_TAU_GRID = ([round(0.01 * step, 3) for step in range(11)]           # 0.00-0.10
                       + [round(0.10 + 0.05 * step, 3) for step in range(1, 19)])  # 0.15-1.00

# --- Search grids, Section 4.1 ---------------------------------------------------------
NEURAL_GRID = [dict(lr=lr, weight_decay=wd)
               for lr, wd in product((1e-4, 3e-4, 1e-3), (1e-4, 1e-3))]
SVM_GRID = [dict(C=value) for value in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)]
# Six configurations per architecture/regime; SVM has its own six-value grid.
assert len(NEURAL_GRID) == len(SVM_GRID) == 6

FAMILIES = ("hog_svm", "cnn", "resnet")
NEURAL_FAMILIES = ("cnn", "resnet")

# One arm per thing that gets scored. The neural families appear once per regime; the SVM
# has no sampler to vary, so it appears once and acts as the control for the comparison --
# its row must not move between the regimes, and Section 5 checks that it does not. The
# offset arm reuses the same fitted SVM and differs only at inference.
ARMS = ("hog_svm", "hog_svm_offset") + tuple(
    f"{family}_{regime}" for family in NEURAL_FAMILIES for regime in IMBALANCE_REGIMES)
ARM_FAMILY = {"hog_svm": "hog_svm", "hog_svm_offset": "hog_svm",
              **{f"{f}_{r}": f for f in NEURAL_FAMILIES for r in IMBALANCE_REGIMES}}
ARM_REGIME = {f"{f}_{r}": r for f in NEURAL_FAMILIES for r in IMBALANCE_REGIMES}

# --- Tail-aware reporting buckets, Section 5.2 -----------------------------------------
SUPPORT_BUCKETS = [
    ("head (>=1000)", 1000, np.inf),
    ("body (100-999)", 100, 1000),
    ("tail (10-99)", 10, 100),
    ("rare (<10)", 0, 10),
]

if QUICK_RUN:
    SEARCH_EPOCHS, CONFIRM_EPOCHS, WARMUP_EPOCHS, PATIENCE = 1, 2, 1, 2
    ALLOW_CPU = True     # a structural check is allowed to be slow; it must not be blocked

print(f"QUICK_RUN = {QUICK_RUN} | search {SEARCH_EPOCHS} epochs, confirm {CONFIRM_EPOCHS}")
print(f"Grids: {len(SVM_GRID)} SVM arms and {len(NEURAL_GRID)} neural arms per architecture/regime (24 neural trials total)")

def find_repository_root():
    """The nearest ancestor directory holding the audited manifest."""
    marker = Path("preprocessed_datasets") / "train_manifest.csv"
    here = Path.cwd().resolve()
    for parent in (here, *here.parents):
        if (parent / marker).is_file():
            return parent
    raise FileNotFoundError(
        f"No {marker} in {here} or any of its parents. Run 00_eda_and_preprocessing.ipynb "
        "first, since its Section 3.4 writes the manifest, and run this notebook from "
        "inside the assignment repository."
    )


REPO_ROOT = find_repository_root()
MANIFEST = REPO_ROOT / "preprocessed_datasets" / "train_manifest.csv"
TRAIN_IMAGE_DIR = REPO_ROOT / "datasets" / "train" / "images_train"
TEST_IMAGE_DIR = REPO_ROOT / "datasets" / "test" / "images_test"
PREDICTION_TEMPLATE = REPO_ROOT / "datasets" / "test" / "styles_prediction.csv"

# One directory holds everything this notebook writes, grouped by what each file is rather
# than by the section that happens to produce it: final/ is what you deploy, tables/ is what
# the report quotes, figures/ is what it prints, predictions/ is what gets submitted.
OUTPUT_DIR = RUN_ROOT / "models" / ("task1" if RANK == 0 else f"_rank{RANK}")
MODEL_DIR = OUTPUT_DIR / "final"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
for _directory in (MODEL_DIR, TABLE_DIR, FIGURE_DIR, PREDICTION_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

for _required in (MANIFEST, TRAIN_IMAGE_DIR, TEST_IMAGE_DIR, PREDICTION_TEMPLATE):
    if not _required.exists():
        raise FileNotFoundError(f"Required input is missing: {_required}")

print("Repository root :", REPO_ROOT)
print("Manifest        :", MANIFEST.relative_to(REPO_ROOT))
print("Writes to       :", OUTPUT_DIR.relative_to(REPO_ROOT))


def save_figure(name):
    """Write the current figure under a registered name.

    Call this immediately BEFORE plt.show(). In the inline backend show() closes the
    figure, so a savefig placed after it writes a blank page: a quiet failure rather than
    an error.
    """
    plt.savefig(FIGURE_DIR / f"{name}.png", dpi=200, bbox_inches="tight")

def choose_device():
    """CUDA where it exists, then Apple Silicon's Metal backend, then CPU.

    MPS is checked through getattr because the attribute is absent on torch builds older
    than 1.12 rather than merely reporting False, and an AttributeError here would be a
    confusing way to learn that.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{LOCAL_RANK}")
    metal = getattr(torch.backends, "mps", None)
    if metal is not None and metal.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = choose_device()

if DEVICE.type == "cuda":
    _name = torch.cuda.get_device_name(DEVICE)
    _total = torch.cuda.get_device_properties(DEVICE).total_memory / 1e9
    _capability = torch.cuda.get_device_capability(DEVICE)
    print(f"GPU: {_name} | {_total:.1f} GB | compute capability {_capability[0]}.{_capability[1]}")
    print(f"torch {torch.__version__} built against CUDA {torch.version.cuda}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True   # autotunes convolutions for this fixed input size
elif DEVICE.type == "cpu" and not ALLOW_CPU:
    raise RuntimeError(
        "No CUDA or MPS device. Training the ResNet on a CPU takes roughly ten minutes per "
        "epoch, which presents as a hang rather than as an error, so the notebook refuses "
        "to start instead. Set ALLOW_CPU = True in Section 1.2 to accept the slowdown "
        "deliberately, or run the HOG + SVM family alone, which never touches the GPU."
    )
else:
    print(f"Device: {DEVICE.type}")

# bf16 has the dynamic range of fp32 and needs no loss scaling. Turing cards such as the T4
# lack it, so fp16 with a gradient scaler is the fallback. Neither changes what is learned.
#
# The capability is read directly rather than through torch.cuda.is_bf16_supported(), which
# grew an `including_emulation=True` default and now answers True on a T4: the card can
# *represent* bf16 by emulating it, at roughly a fifth of the throughput its fp16 tensor
# cores would give, with no error and nothing in the log to say so. Native bf16 starts at
# compute capability 8.0 (Ampere), so that is what gets asked.
AMP_ENABLED = USE_AMP and DEVICE.type == "cuda"

# The GTX 16-series is TU116: Turing with the tensor cores removed. Its fp16 path is not
# reliable, and it fails silently rather than loudly. A BatchNorm output overflows fp16's
# 65504 ceiling, the resulting inf is written into that layer's running statistics during
# the forward pass -- where the gradient scaler cannot skip it, because the scaler only
# guards the optimiser step -- and every evaluation loss from then on is NaN.
#
# The evidence that this is the card and not the arithmetic: with every generator seeded,
# the ResNet trains cleanly when it is the first model in a process and goes NaN on its
# first batch when a CNN was trained before it, so the inputs and the initial weights are
# identical in both cases. It is not cudnn autotuning either, since it fails just as often
# with benchmark disabled. In fp32 it is clean every time. A card with no tensor cores
# gains little from fp16 anyway, so it is refused here rather than left to quietly ruin a
# multi-hour run.
if AMP_ENABLED and "GTX 16" in _name.upper():
    AMP_ENABLED = False
    print("Mixed precision disabled: TU116 (GTX 16-series) fp16 is unreliable here, "
          "so training runs in fp32.")

NATIVE_BF16 = AMP_ENABLED and torch.cuda.get_device_capability(DEVICE)[0] >= 8
AMP_DTYPE = torch.bfloat16 if NATIVE_BF16 else torch.float16
CHANNELS_LAST = DEVICE.type == "cuda"     # NHWC, the layout the tensor-core kernels want
CACHE_ON_DEVICE = DEVICE.type == "cuda"   # hold the uint8 image cache in VRAM: about 0.6 GB

torch.set_num_threads(VISIBLE_CORES)


def set_seed(seed):
    """Seed every generator this notebook draws from.

    Torch, NumPy and Python are all seeded because the augmentation, the weight
    initialisation and the batch order each draw from a different one. Without all three, a
    "same seed" rerun is not actually a rerun.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_STATE)
print(f"Mixed precision: {AMP_ENABLED}"
      f"{'' if not AMP_ENABLED else ' (' + str(AMP_DTYPE).replace('torch.', '') + ')'}"
      f" | channels_last: {CHANNELS_LAST} | image cache on device: {CACHE_ON_DEVICE}")
print("Seeded at", RANDOM_STATE)

def load_manifest(target, image_dir=None):
    """The audited manifest, filtered to rows labelled for one target, with a path column."""
    image_dir = Path(image_dir or TRAIN_IMAGE_DIR)
    frame = pd.read_csv(MANIFEST)
    if target not in frame.columns:
        raise KeyError(f"{target!r} is not a manifest column. Available: {list(frame.columns)}")
    frame = frame.loc[frame[target].notna()].reset_index(drop=True)
    # str rather than Path: a Path survives PIL but writes a machine-local absolute path if
    # a split is ever exported to CSV.
    frame["path"] = frame["filename"].map(lambda name: str(image_dir / name))
    return frame


catalogue = load_manifest(TARGET)
print(f"{len(catalogue):,} labelled images | {catalogue[TARGET].nunique()} article types")
display(catalogue[["id", TARGET, "gender", "season", "usage"]].head())

class_counts = catalogue[TARGET].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
class_counts.head(20).plot.bar(ax=axes[0], color=PALETTE[0])
axes[0].set(title="20 largest classes", ylabel="Images", xlabel="")
axes[0].tick_params(axis="x", labelrotation=75, labelsize=8)

# Log scale, because a linear axis over a 5,000:1 range renders the entire tail as zero.
axes[1].plot(range(len(class_counts)), class_counts.to_numpy(), color=PALETTE[1])
axes[1].set(title="Every class, ranked", xlabel="Class rank", ylabel="Images", yscale="log")
for threshold in (1000, 100, 10):
    axes[1].axhline(threshold, color=MUTED, lw=0.8, ls="--")

plt.tight_layout()
save_figure("fig01_class_distribution")
plt.show()

# The support buckets are the unit the Section 5 metric breakdown reports in, so the class
# counts behind them are established here rather than asserted there.
bucket_summary = pd.DataFrame([
    {"Bucket": name,
     "Classes": int(((class_counts >= low) & (class_counts < high)).sum()),
     "Images": int(class_counts[(class_counts >= low) & (class_counts < high)].sum())}
    for name, low, high in SUPPORT_BUCKETS
])
bucket_summary["Share of images %"] = (
    bucket_summary["Images"] / len(catalogue) * 100).round(1)
display(bucket_summary)
print(f"Largest class : {class_counts.index[0]} at {class_counts.iloc[0]:,} images")
print(f"Smallest class: {class_counts.iloc[-1]} image(s)")
print(f"Imbalance ratio: {class_counts.iloc[0] / class_counts.iloc[-1]:,.0f} : 1")

def make_split(frame, target, validation_share, random_state):
    """Split a task frame into training and validation rows, stratified, grouped.

    Returns:
        (training frame, validation frame), each with a fresh index.
    """
    frame = frame.loc[frame[target].notna()].reset_index(drop=True)
    assert frame["group_id"].is_unique, (
        "Multi-row group_id values found. Section 3.2 of notebook 00 should have collapsed "
        "every duplicate group; re-run it before splitting."
    )

    group_label = frame.groupby("group_id")[target].agg(
        lambda values: values.value_counts().index[0])
    groups_per_class = group_label.value_counts()
    # A class with one group cannot be split, so it goes wholly to training.
    splittable = group_label[~group_label.isin(groups_per_class[groups_per_class < 2].index)]

    _, validation_groups = train_test_split(
        splittable.index, test_size=validation_share,
        stratify=splittable.values, random_state=random_state)

    is_validation = frame["group_id"].isin(set(validation_groups))
    return (frame.loc[~is_validation].reset_index(drop=True),
            frame.loc[is_validation].reset_index(drop=True))


def check_disjoint(*frames):
    """Fail loudly if any id or group_id appears in more than one split."""
    for left, right in combinations(frames, 2):
        for column in ("id", "group_id"):
            overlap = set(left[column].astype(str)) & set(right[column].astype(str))
            assert not overlap, f"Leakage across splits on {column}: {sorted(overlap)[:5]}"

# Reporting is drawn first, off the whole manifest, and then never touched again until
# Section 5. Tuning is drawn from what is left, with a different seed so the two draws are
# independent rather than nested copies of one ordering.
labelled_frame, report_frame = make_split(
    catalogue, TARGET, REPORTING_SHARE, RANDOM_STATE)
fit_frame, tune_frame = make_split(
    labelled_frame, TARGET, TUNING_SHARE, RANDOM_STATE + 1)

check_disjoint(fit_frame, tune_frame, report_frame)

if QUICK_RUN:
    # Keep every class present in training, even in a structural run.
    fit_frame = fit_frame.groupby(TARGET, group_keys=False).head(5).reset_index(drop=True)
    tune_frame = tune_frame.groupby(TARGET, group_keys=False).head(2).reset_index(drop=True)
    report_frame = report_frame.groupby(TARGET, group_keys=False).head(2).reset_index(drop=True)
    labelled_frame = pd.concat([fit_frame, tune_frame], ignore_index=True)

# The class order is fixed here, once, and every model, table and prediction below uses it.
# Sorting rather than taking pandas' encounter order makes the mapping reproducible.
CLASSES = sorted(labelled_frame[TARGET].unique())
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)
CLASS_SUPPORT = np.array([(labelled_frame[TARGET] == name).sum() for name in CLASSES])

# A class the models can never predict would silently cost macro-F1 on every split.
assert set(CLASSES) == set(fit_frame[TARGET]), "A class is absent from the fitting rows."

split_summary = pd.DataFrame([
    {"Split": name, "Rows": len(frame),
     "Share %": round(len(frame) / len(catalogue) * 100, 1),
     "Classes present": frame[TARGET].nunique()}
    for name, frame in (("fit", fit_frame), ("tuning", tune_frame), ("reporting", report_frame))
])
display(split_summary)

for _name, _frame in (("fit", fit_frame), ("tuning", tune_frame), ("reporting", report_frame)):
    _frame[["id", TARGET, "group_id"]].to_csv(TABLE_DIR / f"split_{_name}.csv", index=False)
print("Split membership written to", TABLE_DIR.name + "/split_*.csv")

def standardize_image(image, target_size=IMAGE_TARGET_SIZE):
    """The deterministic input transform. Augmentation comes after, normalisation last."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    if image.size == tuple(target_size):
        return image                      # already the target size: no resampling at all
    return ImageOps.pad(image, tuple(target_size), method=Image.Resampling.BILINEAR,
                        color=IMAGE_PAD_RGB, centering=(0.5, 0.5))


@shared_cpu
def build_image_cache(frame, description):
    """Decode a frame's images once into one uint8 array, in the frame's row order.

    Returns:
        Array of shape (rows, height, width, 3), dtype uint8.
    """
    width, height = IMAGE_TARGET_SIZE
    images = np.empty((len(frame), height, width, 3), dtype=np.uint8)
    start = time.time()
    for position, path in enumerate(frame["path"]):
        with Image.open(path) as handle:
            images[position] = np.asarray(standardize_image(handle))
        if position and position % 10000 == 0:
            print(f"  {description}: {position:,} / {len(frame):,}")
    print(f"{description}: {len(frame):,} images in {time.time() - start:.0f}s "
          f"({images.nbytes / 1e6:.0f} MB)")
    return images


fit_images = build_image_cache(fit_frame, "fitting")
tune_images = build_image_cache(tune_frame, "tuning")

def to_indices(frame):
    """Class indices for a frame, in the fixed CLASSES order."""
    mapped = frame[TARGET].map(CLASS_TO_INDEX)
    assert mapped.notna().all(), "A row carries a class absent from CLASSES."
    return mapped.to_numpy(dtype=np.int64)


y_fit, y_tune = to_indices(fit_frame), to_indices(tune_frame)
# The classes actually present in tuning. Scoring against these rather than against all 124
# is what keeps a class that cannot appear from being counted as a zero.
SCOREABLE = np.unique(y_tune)
print(f"y_fit {y_fit.shape} | y_tune {y_tune.shape} | scoreable classes {len(SCOREABLE)}")

def normalisation_from(images):
    """Per-channel (mean, std) over a uint8 image cache, on the [0, 1] scale."""
    total, squares, n_pixels = np.zeros(3), np.zeros(3), 0
    for begin in range(0, len(images), 1024):
        block = images[begin:begin + 1024].astype(np.float32) / 255.0
        total += block.sum((0, 1, 2), dtype=np.float64)
        squares += np.einsum("nhwc,nhwc->c", block, block, dtype=np.float64)
        n_pixels += int(np.prod(block.shape[:3]))
    mean = total / n_pixels
    # Clamped away from zero: a constant channel would otherwise divide by zero.
    std = np.maximum(np.sqrt(np.maximum(squares / n_pixels - mean ** 2, 0)), 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def set_normalisation(mean, std):
    """Publish the constants as both arrays and broadcastable device tensors."""
    global NORM_MEAN, NORM_STD, NORM_MEAN_T, NORM_STD_T
    NORM_MEAN, NORM_STD = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    NORM_MEAN_T = torch.tensor(NORM_MEAN, device=DEVICE).view(1, 3, 1, 1)
    NORM_STD_T = torch.tensor(NORM_STD, device=DEVICE).view(1, 3, 1, 1)


set_normalisation(*normalisation_from(fit_images))
print("Mean (R, G, B):", np.round(NORM_MEAN, 4))
print("Std  (R, G, B):", np.round(NORM_STD, 4))

LUMA = torch.tensor([0.299, 0.587, 0.114], device=DEVICE).view(1, 3, 1, 1)


def augment_batch(x):
    """Apply the Section 2.4 policy to a batch, with independent parameters per sample.

    Args:
        x: float tensor of shape (n, 3, height, width) with values in [0, 1].

    Returns:
        A tensor of the same shape and range.
    """
    n = x.shape[0]
    device = x.device

    # torch.where selects per sample, so the flip is genuinely drawn per image.
    flip = torch.rand(n, device=device) < AUG_FLIP_PROBABILITY
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)

    # Rotation and translation in one affine warp. The height/width factors correct for
    # affine_grid's normalised coordinates, without which the rotation would shear.
    height, width = x.shape[-2], x.shape[-1]
    angle = (torch.rand(n, device=device) * 2 - 1) * math.radians(AUG_ROTATION_DEGREES)
    shift_x = (torch.rand(n, device=device) * 2 - 1) * AUG_TRANSLATE_FRACTION * 2
    shift_y = (torch.rand(n, device=device) * 2 - 1) * AUG_TRANSLATE_FRACTION * 2
    cos, sin = torch.cos(angle), torch.sin(angle)

    theta = torch.zeros(n, 2, 3, device=device)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin * height / width
    theta[:, 0, 2] = shift_x
    theta[:, 1, 0] = sin * width / height
    theta[:, 1, 1] = cos
    theta[:, 1, 2] = shift_y

    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    # Offset by -1 so grid_sample's zero padding lands on white once shifted back. Rotating
    # a white-background product into a black border would be a stronger signal than the
    # rotation itself.
    x = F.grid_sample(x - 1.0, grid, mode="bilinear", padding_mode="zeros",
                      align_corners=False) + 1.0

    def factor():
        return 1.0 + (torch.rand(n, 1, 1, 1, device=device) * 2 - 1) * AUG_JITTER_STRENGTH

    x = x * factor()                                       # brightness
    grey = (x * LUMA).sum(dim=1, keepdim=True)
    x = (x - grey) * factor() + grey                       # saturation
    mean = grey.mean(dim=(2, 3), keepdim=True)
    x = (x - mean) * factor() + mean                       # contrast

    return x.clamp_(0.0, 1.0)

# What the network actually sees. Eight draws of one image, so the spread of the policy is
# visible rather than described.
_sample = torch.from_numpy(fit_images[:1].copy()).to(DEVICE)
_sample = _sample.permute(0, 3, 1, 2).float().div_(255.0).repeat(8, 1, 1, 1)
set_seed(RANDOM_STATE)
_augmented = augment_batch(_sample).cpu().permute(0, 2, 3, 1).numpy()

fig, axes = plt.subplots(1, 9, figsize=(13, 2.2))
axes[0].imshow(fit_images[0])
axes[0].set_title("original", fontsize=8)
for index, ax in enumerate(axes[1:]):
    ax.imshow(_augmented[index])
    ax.set_title(f"draw {index + 1}", fontsize=8)
for ax in axes:
    ax.axis("off")
fig.suptitle(f"Augmentation policy applied to one {fit_frame[TARGET].iloc[0]} image", y=1.06)
plt.tight_layout()
save_figure("fig02_augmentation_preview")
plt.show()

HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    feature_vector=True,
)

_HOG_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _hog_chunk(chunk):
    """HOG descriptors for a contiguous block of images, in the block's own order.

    Module-level rather than a closure so joblib's process backend can pickle it. Chunked
    rather than one task per image, because 30,000 tasks would spend more time in dispatch
    than in the descriptor.
    """
    features = None
    for position, image in enumerate(chunk):
        grey = (image.astype(np.float32) @ _HOG_LUMA) / 255.0
        descriptor = hog(grey, **HOG_PARAMS).astype(np.float32)
        if features is None:
            features = np.empty((len(chunk), descriptor.size), dtype=np.float32)
        features[position] = descriptor
    return features


@shared_cpu
def hog_features(images, description, chunk_size=2048):
    """HOG descriptor per image, computed on the luminance channel.

    Returns:
        float32 array of shape (rows, n_features).
    """
    start = time.time()
    bounds = list(range(0, len(images), chunk_size))
    parallel = VISIBLE_CORES > 1 and len(bounds) > 1

    if parallel:
        # A failure here is a throughput problem, not a correctness one, so it falls back
        # rather than taking the cell down with it.
        try:
            blocks = joblib.Parallel(n_jobs=VISIBLE_CORES, prefer="processes")(
                joblib.delayed(_hog_chunk)(images[begin:begin + chunk_size])
                for begin in bounds)
        except Exception as error:                 # noqa: BLE001 - any failure falls back
            print(f"  parallel HOG unavailable, computing serially: {error}")
            parallel = False
    if not parallel:
        blocks = [_hog_chunk(images[begin:begin + chunk_size]) for begin in bounds]

    features = np.concatenate(blocks, axis=0)
    print(f"{description}: {features.shape[0]:,} x {features.shape[1]} features in "
          f"{time.time() - start:.0f}s ({features.nbytes / 1e6:.0f} MB, "
          f"{'parallel' if parallel else 'serial'})")
    return features


@shared_cpu
def svm_fit(x, y, config):
    """A linear SVM on HOG features, with balanced class weights.

    ConvergenceWarning is promoted to an error. A non-converged SVM that quietly scores a
    little worse would enter model selection as a weaker competitor, and the comparison
    would be reporting an optimisation failure as a property of the model family.
    """
    model = LinearSVC(**config, dual=False, class_weight="balanced",
                      max_iter=20000, random_state=RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y)
    assert np.array_equal(model.classes_, np.arange(N_CLASSES))
    return model

class PlainCNN(nn.Module):
    """VGG-style stack sized for 60x80 inputs. The reference architecture."""

    def __init__(self, n_classes=None, width=32, dropout=0.3):
        super().__init__()
        n_classes = n_classes or N_CLASSES

        def block(in_channels, out_channels, pool=True):
            layers = [
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(3, width),                 # 80x60 -> 40x30
            block(width, width * 2),         # 40x30 -> 20x15
            block(width * 2, width * 4),     # 20x15 -> 10x7
            block(width * 4, width * 8, pool=False),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(width * 8, n_classes)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.dropout(x))

class BasicBlock(nn.Module):
    """Standard two-convolution residual block with an optional projection shortcut."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # A shortcut can only be added to the block output if both match in shape, so a
        # change of stride or width needs a 1x1 projection to bring it into line.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class SmallResNet(nn.Module):
    """ResNet-18 topology with a stride-1 3x3 stem, sized for 60x80 inputs."""

    def __init__(self, n_classes=None, width=64, blocks=(2, 2, 2, 2), dropout=0.3):
        super().__init__()
        n_classes = n_classes or N_CLASSES
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
        )
        stages, in_channels = [], width
        for stage_index, n_blocks in enumerate(blocks):
            out_channels = width * (2 ** stage_index)
            for block_index in range(n_blocks):
                # Downsample once per stage, at its first block, and never in stage 0.
                stride = 2 if (block_index == 0 and stage_index > 0) else 1
                stages.append(BasicBlock(in_channels, out_channels, stride))
                in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, n_classes)

    def forward(self, x):
        x = self.pool(self.stages(self.stem(x))).flatten(1)
        return self.fc(self.dropout(x))


FACTORIES = {"cnn": PlainCNN, "resnet": SmallResNet}

for _family, _factory in FACTORIES.items():
    _model = _factory()
    print(f"{_family:>7}: {sum(p.numel() for p in _model.parameters()):>10,} parameters")
    del _model

class BatchStream:
    """Device-resident batches, replacing Dataset plus DataLoader.

    The images are held once as uint8 in the device's memory and converted to float per
    batch, so nothing is copied from the host during training and the host holds no
    per-batch buffers. At this image size that removes the dataloader from the critical
    path entirely.

    Args:
        images: uint8 array of shape (rows, height, width, 3), or an existing device
            tensor to share with another stream.
        labels: integer class indices aligned to `images`.
        batch_size: rows per batch.
        augment: apply the Section 2.4 policy. Training only.
        shuffle: reorder each epoch.
        sampler: "instance" draws every row once per epoch, "balanced" draws rows with
            probability 1/count so the classes arrive in roughly equal numbers.
        drop_last: discard a final batch shorter than `batch_size`. Training only; see
            `__iter__` for why a short batch is not merely wasteful but unsafe.
    """

    def __init__(self, images, labels, batch_size=BATCH_SIZE, augment=False, shuffle=False,
                 sampler="instance", drop_last=False):
        if torch.is_tensor(images):
            self.images = images                    # shared with another stream, not copied
        else:
            self.images = torch.from_numpy(writable_array(images))
            if CACHE_ON_DEVICE:
                self.images = self.images.to(DEVICE, non_blocking=True)
        self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=DEVICE)
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.shard_eval = False

        assert sampler in ("instance", "balanced"), sampler
        self.sampler = sampler
        if sampler == "balanced":
            counts = torch.bincount(self.labels, minlength=N_CLASSES).float()
            # Weight per row, not per class: multinomial draws over rows, so every row of a
            # rare class has to carry that class's full share.
            self.sample_weights = (1.0 / counts.clamp(min=1)) ** RESAMPLE_POWER
            self.sample_weights = self.sample_weights[self.labels]
        else:
            self.sample_weights = None

    def __len__(self):
        if self.drop_last and len(self.labels) >= self.batch_size:
            return len(self.labels) // self.batch_size
        return math.ceil(len(self.labels) / self.batch_size)

    def __iter__(self):
        if self.sampler == "balanced":
            # With replacement, and the epoch keeps its length: this changes which rows the
            # arm sees, not how many gradient steps it gets, so the budget stays equal to
            # the other regime's. A one-image class has to be drawn many times over for the
            # balance to hold at all, and each draw is augmented independently.
            order = torch.multinomial(self.sample_weights, len(self.labels),
                                      replacement=True).to(self.images.device)
        elif self.shuffle:
            order = torch.randperm(len(self.labels), device=self.images.device)
        else:
            order = torch.arange(len(self.labels), device=self.images.device)
        if DDP_ACTIVE and self.augment:
            dist.broadcast(order, src=0)  # one shared global sampling order
        if DDP_ACTIVE and self.shard_eval:
            order = order[LOCAL_RANK::WORLD_SIZE]
        limit = len(order)
        if self.drop_last and limit >= self.batch_size:
            # A short final batch is not just a smaller gradient step. BatchNorm estimates
            # each channel's variance from the batch, and on few rows that estimate can
            # collapse toward zero in the deeper stages, which sends the normalised
            # activations past fp16's 65504 ceiling. The resulting inf is written into the
            # running statistics, and those are updated in the forward pass, so the
            # gradient scaler skipping the step does not undo it: the model is poisoned for
            # the rest of training and every later eval-mode loss is NaN. Dropping the
            # remainder costs at most one batch of an epoch and removes the failure mode.
            limit -= limit % self.batch_size
        for start in range(0, limit, self.batch_size):
            index = order[start:start + self.batch_size]
            if DDP_ACTIVE and self.augment:
                # Pad only the last global batch, as DistributedSampler does.
                padding = (-len(index)) % WORLD_SIZE
                if padding:
                    index = torch.cat([index, index[:1].repeat(padding)])
                index = index[LOCAL_RANK::WORLD_SIZE]
            batch = self.images[index].to(DEVICE, non_blocking=True)
            x = batch.permute(0, 3, 1, 2).float().div_(255.0)
            if self.augment:
                x = augment_batch(x)
            # Normalisation is applied last, after augmentation, so the jitter operates on
            # the [0, 1] scale the policy was designed against.
            x = (x - NORM_MEAN_T) / NORM_STD_T
            if CHANNELS_LAST:
                x = x.contiguous(memory_format=torch.channels_last)
            yield x, self.labels[index.to(self.labels.device)]

def make_criterion(y, regime):
    """The loss for one imbalance regime.

    Under "reweight", weighting each class by n / (K * count) makes the 1-image classes
    contribute as much total loss as Tshirts does. Without it a model reaches respectable
    accuracy by learning the eight head classes and answering them for everything, which is
    exactly the failure macro-F1 is chosen to expose.

    Under "resample" the same criterion is returned unweighted, because that regime's
    BatchStream has already equalised the classes by drawing them equally often. Weighting
    on top of balanced sampling would apply 1/count twice.
    """
    assert regime in IMBALANCE_REGIMES, regime
    if regime == "resample":
        def criterion(logits, targets):
            return F.cross_entropy(logits, targets, label_smoothing=LABEL_SMOOTHING)
        return criterion

    counts = np.bincount(y, minlength=N_CLASSES)
    assert (counts > 0).all(), "A class has no fitting rows; the weight would be infinite."
    weights = torch.as_tensor(len(y) / (N_CLASSES * counts), device=DEVICE, dtype=torch.float32)

    def criterion(logits, targets):
        losses = F.cross_entropy(logits, targets, reduction="none",
                                 label_smoothing=LABEL_SMOOTHING)
        # Mean over rows rather than a weighted mean, so the loss scale does not drift with
        # whichever classes a batch happened to draw.
        return (losses * weights[targets]).mean()

    return criterion


def balanced_criterion(y):
    """The inverse-frequency criterion under the name the original recipe used."""
    return make_criterion(y, "reweight")


def score_predictions(y_true, y_pred):
    """Macro-F1, accuracy and weighted F1 over the classes actually present in y_true."""
    assert len(y_true) == len(y_pred) and len(y_true)
    present = np.unique(y_true)
    return dict(
        macro_f1=float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        weighted_f1=float(f1_score(y_true, y_pred, labels=present, average="weighted", zero_division=0)),
    )


def choose_best(rows):
    """The winning row under the selection rule, declared once and applied everywhere.

    Highest macro-F1; ties broken by accuracy, then by the candidate identifier. The final
    tie-break is lexical rather than arbitrary so that the rule is deterministic: two arms
    that score identically must not select differently on a rerun.
    """
    assert rows and all(np.isfinite(row["macro_f1"]) for row in rows)
    return sorted(rows, key=lambda row: (-row["macro_f1"], -row["accuracy"], row["candidate"]))[0]


def prepare_model(model):
    """Move a model to the device in the memory layout the convolution kernels prefer."""
    model = model.to(DEVICE)
    if CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    if DDP_ACTIVE:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[LOCAL_RANK],
                                        output_device=LOCAL_RANK,
                                        gradient_as_bucket_view=True)
    return model


def make_scaler():
    """A gradient scaler, enabled only for fp16. bf16 has fp32's range and needs no scaling."""
    enabled = AMP_ENABLED and AMP_DTYPE == torch.float16
    try:
        return torch.amp.GradScaler(DEVICE.type, enabled=enabled)
    except (AttributeError, TypeError):     # torch < 2.4 keeps it under torch.cuda.amp
        return torch.cuda.amp.GradScaler(enabled=enabled)


def release(*objects):
    """Drop references the notebook no longer reads and return their VRAM."""
    del objects
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

def run_epoch(model, stream, criterion, optimiser=None, scaler=None):
    """One pass over a stream. Trains if an optimiser is given, otherwise evaluates.

    Returns:
        (mean loss, logits array or None). Logits come back in float32 whatever the
        autocast dtype was, so every metric downstream sees the same precision.

    The running loss stays on the device until the pass ends. Calling .item() per batch
    forces a host synchronisation on every step, and at this model size the step is short
    enough for that stall to be a real share of the epoch. One synchronisation per epoch
    reports the identical number.
    """
    training = optimiser is not None
    model.train(training)
    if DDP_ACTIVE and not training:
        return distributed_evaluate(model, stream, criterion)
    loss_sum = torch.zeros((), device=DEVICE, dtype=torch.float32)
    n_seen, collected = 0, []

    with torch.set_grad_enabled(training):
        for images, targets in stream:
            with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=AMP_ENABLED):
                logits = model(images)
                loss = criterion(logits, targets)

            finite = torch.isfinite(loss).to(torch.int32)
            if DDP_ACTIVE:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite:
                raise FloatingPointError("Non-finite loss. Restart with USE_AMP=False; "
                                         "this run must not be reported.")

            if training:
                optimiser.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    loss.backward()
                    optimiser.step()
            else:
                collected.append(logits.detach().float())

            loss_sum += loss.detach().float() * targets.shape[0]
            n_seen += targets.shape[0]

    if DDP_ACTIVE and training:
        totals = torch.stack([loss_sum, loss_sum.new_tensor(n_seen)])
        dist.all_reduce(totals)
        loss_sum, n_seen = totals[0], totals[1].item()
    mean_loss = (loss_sum / max(n_seen, 1)).item()
    return mean_loss, (torch.cat(collected).cpu().numpy() if collected else None)


def train(model, train_stream, val_stream, y_val, epochs, lr, weight_decay,
          label, patience=PATIENCE, scoreable=None, regime="reweight"):
    """Train with warmup and cosine decay, early stopping on validation macro-F1.

    The best epoch's weights are restored before returning, so the model handed back is the
    model whose score is reported rather than whatever the last epoch happened to leave.

    `regime` selects the imbalance handling, and it has to agree with the sampler the
    training stream was built with: the criterion is weighted only when the stream is not
    balanced. Passing a balanced stream with regime="reweight" would correct twice.

    Returns:
        (history DataFrame, best validation logits, best macro-F1, best epoch).
    """
    scoreable = SCOREABLE if scoreable is None else scoreable
    # The weights come from the rows this call actually trains on, so the same function
    # serves the search (fitting rows) and the Section 5 refit (fitting + tuning rows).
    criterion = make_criterion(train_stream.labels.cpu().numpy(), regime)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = make_scaler()

    def schedule(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / max(WARMUP_EPOCHS, 1)
        progress = (epoch - WARMUP_EPOCHS) / max(epochs - WARMUP_EPOCHS, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule)
    history, best = [], {"macro_f1": -1.0, "epoch": -1, "state": None, "logits": None}
    start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        # Read before the scheduler advances. param_groups holds the NEXT epoch's rate once
        # scheduler.step() has run, so reading it after the step records every epoch against
        # the rate the following epoch would use.
        epoch_lr = optimiser.param_groups[0]["lr"]
        train_loss, _ = run_epoch(model, train_stream, criterion, optimiser, scaler)
        val_loss, val_logits = run_epoch(model, val_stream, criterion)
        scheduler.step()

        val_pred = val_logits.argmax(axis=1)
        macro = f1_score(y_val, val_pred, labels=scoreable, average="macro", zero_division=0)
        history.append({"epoch": epoch + 1, "train loss": train_loss, "val loss": val_loss,
                        "val accuracy": accuracy_score(y_val, val_pred),
                        "val macro-F1": macro, "lr": epoch_lr,
                        "seconds": time.time() - epoch_start})

        if macro > best["macro_f1"]:
            best.update({"macro_f1": macro, "epoch": epoch + 1, "logits": val_logits,
                         "state": {k: v.detach().cpu().clone()
                                   for k, v in unwrap_model(model).state_dict().items()}})

        print(f"  [{label}] epoch {epoch + 1:>3}/{epochs}  train {train_loss:.3f}  "
              f"val {val_loss:.3f}  acc {history[-1]['val accuracy']:.3f}  "
              f"macro-F1 {macro:.4f}  {history[-1]['seconds']:.0f}s")

        # A non-finite epoch is terminal, not a bad epoch to be trained through. BatchNorm
        # updates its running statistics in the forward pass, so an inf that reaches them
        # is permanent and every later epoch reports NaN too. Stopping here turns three
        # hours of tables quietly full of NaN into one error at the point of failure.
        if not (np.isfinite(train_loss) and np.isfinite(val_loss)):
            raise RuntimeError(
                f"[{label}] epoch {epoch + 1} produced a non-finite loss "
                f"(train {train_loss}, val {val_loss}). This is unrecoverable: the "
                f"BatchNorm running statistics are updated during the forward pass, so "
                f"they are already poisoned. If AMP is on, the usual cause is an fp16 "
                f"overflow -- set AMP_ENABLED = False in Section 1.3 and re-run."
            )

        if epoch + 1 - best["epoch"] >= patience:
            print(f"  [{label}] early stop at epoch {epoch + 1}; best was epoch {best['epoch']}")
            break

    unwrap_model(model).load_state_dict(best["state"])          # report and save the same weights
    print(f"  [{label}] best macro-F1 {best['macro_f1']:.4f} at epoch {best['epoch']} "
          f"({time.time() - start:.0f}s total)")
    return pd.DataFrame(history), best["logits"], best["macro_f1"], best["epoch"]

# Both neural families train against the same streams, built once here rather than per arm:
# the uint8 cache is already on the device and rebuilding it per arm would dominate the
# search. The training stream augments and shuffles; the validation stream does neither.
fit_stream = BatchStream(fit_images, y_fit, BATCH_SIZE, augment=True, shuffle=True,
                         drop_last=True)

# The second regime shares the first stream's device cache rather than uploading a second
# copy of the same 24,223 images; only the draw order differs between them.
fit_streams = {
    "reweight": fit_stream,
    "resample": BatchStream(fit_stream.images, y_fit, BATCH_SIZE, augment=True,
                            shuffle=True, sampler="balanced", drop_last=True),
}
tune_stream = BatchStream(tune_images, y_tune, 512)
print(f"fit stream: {len(fit_stream)} batches of {BATCH_SIZE} | "
      f"tuning stream: {len(tune_stream)} batches of 512")

# What the balanced sampler actually does to an epoch, stated in rows rather than trusted.
_drawn = np.bincount(y_fit[torch.multinomial(
    fit_streams["resample"].sample_weights, len(y_fit), replacement=True).cpu().numpy()],
    minlength=N_CLASSES)
_counts = np.bincount(y_fit, minlength=N_CLASSES)
print(f"balanced draw, rarest class: {_counts.min()} row(s) in the split -> "
      f"~{_drawn[_counts.argmin()]} draws per epoch "
      f"(uniform sampling would give {_counts.min()})")

# The descriptor is computed once per split and reused across all six arms. Recomputing it
# per arm would spend about twenty minutes producing identical features six times.
hog_fit = hog_features(fit_images, "fit HOG")
hog_tune = hog_features(tune_images, "tuning HOG")

svm_rows = []
svm_tune_scores = {}
for index, config in enumerate(SVM_GRID):
    start = time.time()
    model = svm_fit(hog_fit, y_fit, config)
    # LinearSVC decides by argmax over these one-vs-rest margins, so keeping the scores
    # costs one pass instead of two and gives Section 4.6 the surface it shifts. The row
    # below is the identical number model.predict would have produced.
    scores = model.decision_function(hog_tune)
    svm_tune_scores[f"hog_svm_{index}"] = scores
    row = dict(candidate=f"hog_svm_{index}", family="hog_svm", regime=None, config=config,
               epochs=None, **score_predictions(y_tune, scores.argmax(1)))
    svm_rows.append(row)
    print(f"  C={config['C']:<6} macro-F1 {row['macro_f1']:.4f}  "
          f"accuracy {row['accuracy']:.4f}  ({time.time() - start:.0f}s)")
    release(model)

hog_contender = choose_best(svm_rows)
hog_contender["arm"] = "hog_svm"
display(pd.DataFrame(svm_rows).drop(columns="epochs"))
print("Selected HOG/SVM arm:", hog_contender["candidate"], hog_contender["config"])

# Fitted against the search-stage SVM, the one trained on the fitting rows only, so the
# tuning split is still held out of everything the offset has seen. Fitting tau against the
# model refitted in Section 5 would be fitting on that model's own training rows, and the
# offset would read better here than it could ever perform on the reporting split.
fit_log_prior = np.log(np.bincount(y_fit, minlength=N_CLASSES) / len(y_fit))
base_scores = svm_tune_scores[hog_contender["candidate"]]

offset_rows = []
for tau in SVM_OFFSET_TAU_GRID:
    shifted = base_scores - tau * fit_log_prior
    offset_rows.append(dict(tau=tau, **score_predictions(y_tune, shifted.argmax(1))))

offset_table = pd.DataFrame(offset_rows)
best_offset = offset_table.loc[offset_table["macro_f1"].idxmax()]
SVM_OFFSET_TAU = float(best_offset["tau"])
offset_table.to_csv(TABLE_DIR / "svm_offset_tau.csv", index=False)

# The offset arm is the same fitted SVM read differently, so it inherits the contender row
# and overrides only what the shift changes.
hog_offset_contender = dict(hog_contender)
hog_offset_contender.update(candidate="hog_svm_offset", arm="hog_svm_offset",
                            macro_f1=float(best_offset["macro_f1"]),
                            accuracy=float(best_offset["accuracy"]),
                            weighted_f1=float(best_offset["weighted_f1"]))

fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.plot(offset_table["tau"], offset_table["macro_f1"], color=PALETTE[0], label="macro-F1")
ax.plot(offset_table["tau"], offset_table["accuracy"], color=PALETTE[1], ls="--",
        label="accuracy")
ax.axvline(SVM_OFFSET_TAU, color=MUTED, ls=":", label=f"selected tau = {SVM_OFFSET_TAU}")
ax.set(title="Per-class decision offset, tuning split", xlabel="tau", ylabel="Score")
ax.legend(fontsize=8)
plt.tight_layout()
save_figure("fig06_svm_offset_tau")
plt.show()

print(f"tau = {SVM_OFFSET_TAU}: tuning macro-F1 {hog_contender['macro_f1']:.4f} -> "
      f"{best_offset['macro_f1']:.4f} ({best_offset['macro_f1'] - hog_contender['macro_f1']:+.4f}), "
      f"accuracy {hog_contender['accuracy']:.4f} -> {best_offset['accuracy']:.4f}")

def neural_arm(family, config, epochs, label, patience=None, regime="reweight"):
    """Train one arm from scratch and score it on the tuning split.

    Every arm is reseeded identically before it builds its model, so the arms differ by
    their hyperparameters and by nothing else — not by where the global RNG happened to be
    when the previous arm finished.
    """
    set_phase("train:" + label)
    set_seed(RANDOM_STATE)
    model = prepare_model(FACTORIES[family]())
    history, logits, macro, best_epoch = train(
        model, fit_streams[regime], tune_stream, y_tune, epochs=epochs,
        patience=epochs if patience is None else patience, label=label,
        regime=regime, **config)
    history.to_csv(TABLE_DIR / f"history_{label}.csv", index=False)
    row = dict(candidate=label, family=family, regime=regime, config=config,
               epochs=best_epoch, **score_predictions(y_tune, logits.argmax(1)))
    release(model)
    return row, history

search_rows = list(svm_rows)
neural_search = {}

for family in ("cnn", "resnet"):
    for regime in IMBALANCE_REGIMES:
        arm = f"{family}_{regime}"
        print(f"\n=== {arm}: {len(NEURAL_GRID)} arms at {SEARCH_EPOCHS} epochs ===")
        rows = []
        for index, config in enumerate(NEURAL_GRID):
            row, _ = neural_arm(family, config, SEARCH_EPOCHS,
                                f"{arm}_search_{index}", regime=regime)
            rows.append(row)
        neural_search[arm] = rows
        search_rows.extend(rows)
        display(pd.DataFrame(rows))


contenders = [hog_contender, hog_offset_contender]
confirm_histories = {}

for family in NEURAL_FAMILIES:
    for regime in IMBALANCE_REGIMES:
        arm = f"{family}_{regime}"
        selected = choose_best(neural_search[arm])
        print(f"\n=== {arm} confirmation: {selected['config']} at {CONFIRM_EPOCHS} epochs ===")
        # Early stopping is enabled here, unlike in the search: the point of the
        # confirmation run is the best achievable score for this recipe, and the epoch it
        # was reached at is what the Section 5 refit replays.
        row, history = neural_arm(family, selected["config"], CONFIRM_EPOCHS,
                                  f"{arm}_confirm", patience=PATIENCE, regime=regime)
        row["arm"] = arm
        contenders.append(row)
        confirm_histories[arm] = history

candidate_table = pd.DataFrame(contenders)
display(candidate_table)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
# Solid for the original regime, dashed for balanced sampling, one colour per family, so
# the pairing that carries the comparison is the one the eye picks up first.
for (arm, history), colour in zip(confirm_histories.items(), PALETTE):
    style = "-" if arm.endswith("reweight") else "--"
    axes[0].plot(history["epoch"], history["train loss"], color=colour, ls=style, label=arm)
    axes[1].plot(history["epoch"], history["val macro-F1"], color=colour, ls=style, label=arm)
axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy")
axes[1].set(title="Tuning macro-F1", xlabel="Epoch", ylabel="Macro-F1")
axes[1].axhline(hog_contender["macro_f1"], color=MUTED, ls=":", label="HOG + SVM")
for ax in axes:
    ax.legend(fontsize=8)
fig.suptitle("Confirmation runs at the full 40-epoch budget", y=1.03)
plt.tight_layout()
save_figure("fig03_confirmation_curves")
plt.show()

# One rule, applied to every confirmed arm. Everything after this point is reporting: the
# decision is frozen here, before any reporting row has been touched. The imbalance regime
# competes on the same footing as any other choice rather than being assumed to help.
winner = choose_best(contenders)

selection = dict(
    rule="highest tuning macro-F1; ties broken by accuracy, then candidate identifier",
    winner=winner["arm"],
    winner_family=winner["family"],
    contenders=contenders,
    recipes={row["arm"]: dict(config=row["config"], epochs=row["epochs"],
                              regime=row.get("regime")) for row in contenders},
    svm_offset_tau=SVM_OFFSET_TAU,
    tuning_normalisation=[NORM_MEAN.tolist(), NORM_STD.tolist()],
)

assert set(selection["recipes"]) == set(ARMS)
(OUTPUT_DIR / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
pd.DataFrame(search_rows).to_csv(TABLE_DIR / "search_all_models.csv", index=False)
candidate_table.to_csv(TABLE_DIR / "selection_candidates.csv", index=False)

print("Selected arm:", selection["winner"])
for arm, recipe in selection["recipes"].items():
    print(f"  {arm:>16}: {recipe}")

# The tuning caches and streams are finished with. Releasing them here returns roughly a
# gigabyte before Section 5 builds the refit cache, which is what keeps the notebook inside
# a 16 GB card.
del fit_stream, fit_streams, tune_stream, hog_fit, hog_tune, fit_images, tune_images
del svm_tune_scores
release()

refit_frame = labelled_frame.copy()
refit_images = build_image_cache(refit_frame, "refit")
y_refit = to_indices(refit_frame)

# Refitted on the new training rows, and on those rows only: the reporting split is still
# untouched at this point and must stay that way.
set_normalisation(*normalisation_from(refit_images))
print("Refit mean:", np.round(NORM_MEAN, 4), "| std:", np.round(NORM_STD, 4))

refit_stream = BatchStream(refit_images, y_refit, BATCH_SIZE, augment=True,
                           shuffle=True, drop_last=True)

# Both regimes again, sharing the one device cache, exactly as in Section 4.1.
refit_streams = {
    "reweight": refit_stream,
    "resample": BatchStream(refit_stream.images, y_refit, BATCH_SIZE, augment=True,
                            shuffle=True, sampler="balanced", drop_last=True),
}

# The prior the offset arm shifts against, recomputed on the rows the SVM is refitted on.
# Section 4.3 fitted tau against the fitting rows; the shape of the offset is unchanged and
# only the counts move, which is what replaying a selected recipe on more data means.
REFIT_LOG_PRIOR = np.log(np.bincount(y_refit, minlength=N_CLASSES) / len(y_refit))
print(f"Refitting on {len(refit_frame):,} rows "
      f"({len(refit_frame) / len(fit_frame) - 1:+.0%} against the search runs)")

FINAL_PATHS, refit_histories, refit_seconds = {}, {}, {}

for arm in ARMS:
    family, recipe = ARM_FAMILY[arm], selection["recipes"][arm]
    set_phase("refit:" + arm)
    refit_start = time.time()

    if arm == "hog_svm_offset":
        # No training of its own. The offset is applied at inference to the SVM refitted
        # just above, so the two SVM arms differ by tau and by nothing else, and the
        # comparison between them is not confounded by a second fit.
        print(f"\n=== refit {arm}: reuses hog_svm, tau={selection['svm_offset_tau']} ===")
        FINAL_PATHS[arm] = FINAL_PATHS["hog_svm"]
        refit_seconds[arm] = 0.0
        continue

    print(f"\n=== refit {arm}: {recipe['config']} ===")
    if family == "hog_svm":
        path = MODEL_DIR / "hog_svm_final.joblib"
        hog_refit = hog_features(refit_images, "refit HOG")
        model = svm_fit(hog_refit, y_refit, recipe["config"])
        joblib.dump(dict(estimator=model, classes=CLASSES, hog_params=HOG_PARAMS,
                         recipe=recipe), path)
        release(hog_refit, model)
    else:
        path = MODEL_DIR / f"{arm}_final.pt"
        set_seed(RANDOM_STATE)
        model = prepare_model(FACTORIES[family]())
        optimiser = torch.optim.AdamW(model.parameters(), **recipe["config"])
        scaler = make_scaler()
        # The criterion follows the arm's regime, and so does the stream. The two travel
        # together or the correction is applied twice.
        criterion = make_criterion(y_refit, recipe["regime"])
        stream = refit_streams[recipe["regime"]]

        # The schedule is shaped by CONFIRM_EPOCHS, not by the number of epochs actually
        # run. Replaying the selected prefix of the confirmation schedule is what makes this
        # the same recipe; re-normalising the cosine over the shorter run would give the
        # model a different learning rate at every step than the one that was confirmed.
        def refit_schedule(epoch):
            if epoch < WARMUP_EPOCHS:
                return (epoch + 1) / max(WARMUP_EPOCHS, 1)
            return 0.5 * (1 + np.cos(np.pi * (epoch - WARMUP_EPOCHS) /
                                     max(CONFIRM_EPOCHS - WARMUP_EPOCHS, 1)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, refit_schedule)
        history = []
        for epoch in range(recipe["epochs"]):
            start = time.time()
            loss, _ = run_epoch(model, stream, criterion, optimiser, scaler)
            history.append(dict(epoch=epoch + 1, loss=loss,
                                lr=optimiser.param_groups[0]["lr"],
                                seconds=time.time() - start))
            scheduler.step()
            print(f"  [{arm} refit] {epoch + 1}/{recipe['epochs']} "
                  f"loss={loss:.4f} {time.time() - start:.0f}s")

        torch.save(dict(family=family, arm=arm, regime=recipe["regime"],
                        state_dict=unwrap_model(model).state_dict(), classes=CLASSES,
                        recipe=recipe, normalisation_mean=NORM_MEAN.tolist(),
                        normalisation_std=NORM_STD.tolist(),
                        image_target_size=list(IMAGE_TARGET_SIZE)), path)
        refit_histories[arm] = pd.DataFrame(history)
        refit_histories[arm].to_csv(TABLE_DIR / f"refit_{arm}.csv", index=False)
        release(model, optimiser, scaler)

    FINAL_PATHS[arm] = path
    refit_seconds[arm] = time.time() - refit_start
    print(f"  saved -> {path.relative_to(REPO_ROOT)} "
          f"({path.stat().st_size / 1e6:.1f} MB, {refit_seconds[arm]:.0f}s to fit)")

del refit_stream, refit_streams, refit_images
release()

# Training is complete on both ranks. Reporting and exports need only rank zero.
if DDP_ACTIVE:
    dist.barrier()
    dist.destroy_process_group()
    DDP_ACTIVE = False
    if RANK != 0:
        raise SystemExit(0)
    restore_full_cpu()


def load_final(arm):
    """Reload a refitted model from disk, so what is scored is what was saved."""
    if ARM_FAMILY[arm] == "hog_svm":
        return joblib.load(FINAL_PATHS[arm])["estimator"]
    model = prepare_model(FACTORIES[ARM_FAMILY[arm]]())
    unwrap_model(model).load_state_dict(torch.load(FINAL_PATHS[arm], map_location=DEVICE,
                                     weights_only=False)["state_dict"])
    return model.eval()


def svm_scores_to_predictions(arm, scores):
    """Argmax over the SVM margins, shifted by the Section 4.3 offset for the offset arm.

    Subtracting tau * log(prior) raises the margin of a class in proportion to how rare it
    is, so the arm trades head precision for tail recall. tau was fitted on the tuning
    split in Section 4.3 and is replayed here unchanged.
    """
    if arm == "hog_svm_offset":
        scores = scores - selection["svm_offset_tau"] * REFIT_LOG_PRIOR
    return scores.argmax(1).astype(np.int64)


def predict_images(arm, model, images, hog_cache=None):
    """Class indices for a uint8 image cache, under the arm's own inference path.

    `hog_cache` lets the two SVM arms share one descriptor pass over the same images;
    recomputing it per arm would spend a minute producing an identical array.
    """
    if ARM_FAMILY[arm] == "hog_svm":
        features = hog_features(images, "HOG inference") if hog_cache is None else hog_cache
        return svm_scores_to_predictions(arm, model.decision_function(features))
    stream = BatchStream(images, np.zeros(len(images), dtype=np.int64), 512)
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in stream:
            with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=AMP_ENABLED):
                predictions.append(model(x).argmax(1).cpu().numpy())
    return np.concatenate(predictions)


set_phase("reporting")
report_images = build_image_cache(report_frame, "reporting")
y_report = to_indices(report_frame)

report_predictions = report_frame[["id", TARGET, "group_id"]].copy()
report_predictions["true_index"] = y_report
results = []


def score_row(name, predicted, **cost):
    """One row of the comparison: overall scores, support buckets, and what it cost."""
    row = dict(model=name, split="reporting", rows=len(y_report),
               selected=name == selection["winner"],
               **score_predictions(y_report, predicted), **cost)
    # Macro-F1 restricted to the classes in each support bucket, so a model that wins
    # overall by winning only on Tshirts is visible as such.
    for bucket_name, low, high in SUPPORT_BUCKETS:
        bucket = np.intersect1d(
            np.flatnonzero((CLASS_SUPPORT >= low) & (CLASS_SUPPORT < high)),
            np.unique(y_report))
        row[bucket_name] = (float(f1_score(y_report, predicted, labels=bucket,
                                           average="macro", zero_division=0))
                            if len(bucket) else None)
    return row


# One descriptor pass, shared by both SVM arms and timed once. Its cost belongs to each of
# them, so it is added back to their prediction time rather than quietly dropped.
_hog_start = time.time()
report_hog = hog_features(report_images, "reporting HOG")
report_hog_seconds = time.time() - _hog_start

for arm in ARMS:
    model = load_final(arm)
    start = time.time()
    predicted = predict_images(arm, model, report_images, hog_cache=report_hog)
    elapsed = time.time() - start
    if ARM_FAMILY[arm] == "hog_svm":
        elapsed += report_hog_seconds
    report_predictions[arm] = predicted

    row = score_row(arm, predicted,
                    prediction_seconds=elapsed,
                    train_seconds=refit_seconds[arm],
                    model_bytes=FINAL_PATHS[arm].stat().st_size)
    row["family"] = ARM_FAMILY[arm]
    row["regime"] = selection["recipes"][arm].get("regime")
    results.append(row)
    print(f"{arm:>18}: macro-F1 {row['macro_f1']:.4f}  accuracy {row['accuracy']:.4f}  "
          f"({row['prediction_seconds']:.1f}s)")
    release(model)

release(report_hog)

# Two non-learning references, fitted on the same rows and scored on the same images. A
# model's score means nothing without the floor it has to clear, and these two disagree in
# the way that is itself the argument for macro-F1: always answering `Tshirts` is right 17%
# of the time, which sounds like a start until macro-F1 reads it at 0.003.
for name, strategy in (("reference_majority", "most_frequent"),
                       ("reference_random", "stratified")):
    dummy = DummyClassifier(strategy=strategy, random_state=RANDOM_STATE)
    dummy.fit(np.zeros((len(y_refit), 1)), y_refit)      # the labels are all these need
    predicted = dummy.predict(np.zeros((len(y_report), 1))).astype(np.int64)
    report_predictions[name] = predicted

    row = score_row(name, predicted)
    results.append(row)
    print(f"{name:>18}: macro-F1 {row['macro_f1']:.4f}  accuracy {row['accuracy']:.4f}")

result_table = pd.DataFrame(results)

# The SVM has no sampler to vary, so its row is the control for the whole comparison: if
# the two regimes had leaked into each other through the shared caches, the streams or the
# seed, it is the row that would move. It cannot move, so this is a real assertion.
_svm = result_table.loc[result_table["model"] == "hog_svm", "macro_f1"].iloc[0]
print(f"\ncontrol: hog_svm macro-F1 {_svm:.4f} -- unaffected by the sampling regime "
      f"by construction")
result_table.to_csv(TABLE_DIR / "task1_results.csv", index=False)
report_predictions.to_csv(PREDICTION_DIR / "reporting_all_models.csv", index=False)
display(result_table)


fig, axes = plt.subplots(1, 2, figsize=(13, 4))
result_table.set_index("model")[["macro_f1", "accuracy", "weighted_f1"]].plot.bar(
    ax=axes[0], color=PALETTE[:3])
axes[0].set(title="Reporting split, overall", ylabel="Score", xlabel="")
axes[0].tick_params(axis="x", labelrotation=0)

# The right panel keeps to the three models. Both references sit within a hair of zero in
# every bucket, so plotting them there would flatten the curves this panel exists to show;
# the left panel is where the floor belongs.
bucket_names = [name for name, _, _ in SUPPORT_BUCKETS]
arms_only = result_table[result_table["model"].isin(ARMS)].set_index("model")
arms_only[bucket_names].T.plot(ax=axes[1], marker="o", color=PALETTE[:len(ARMS)])
axes[1].set(title="Macro-F1 by class support", ylabel="Macro-F1", xlabel="")
axes[1].tick_params(axis="x", labelrotation=20)

plt.tight_layout()
save_figure("fig04_reporting_comparison")
plt.show()

def paired_interval(y_true, predictions_a, predictions_b, n_resamples=1000):
    """95% bootstrap interval for the macro-F1 difference between two models.

    Paired and stratified: both models are scored on the same resample, drawn within class,
    so the interval measures the difference between the models rather than the variance of
    the split, and no resample can drop a rare class entirely.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    groups = [np.flatnonzero(y_true == label) for label in np.unique(y_true)]
    differences = []
    for _ in range(n_resamples):
        index = np.concatenate([rng.choice(group, len(group), replace=True)
                                for group in groups])
        differences.append(score_predictions(y_true[index], predictions_a[index])["macro_f1"] -
                           score_predictions(y_true[index], predictions_b[index])["macro_f1"])
    return np.quantile(differences, [0.025, 0.975]).tolist()


# Every arm against the winner, rather than all fifteen pairs: the question each interval
# has to answer is whether the margin over the arm that was actually selected is real. The
# same-family pairs that isolate the regime are in there by construction.
intervals = []
for other in [arm for arm in ARMS if arm != selection["winner"]]:
    low, high = paired_interval(report_predictions["true_index"].to_numpy(),
                                report_predictions[selection["winner"]].to_numpy(),
                                report_predictions[other].to_numpy(),
                                n_resamples=100 if QUICK_RUN else 1000)
    intervals.append(dict(model_a=selection["winner"], model_b=other,
                          lower_95=low, upper_95=high))
    print(f"{selection['winner']} - {other}: [{low:+.4f}, {high:+.4f}]"
          f"{'  (excludes zero)' if low > 0 or high < 0 else '  (includes zero)'}")

interval_table = pd.DataFrame(intervals)
interval_table.to_csv(TABLE_DIR / "paired_holdout_intervals.csv", index=False)

class_names = np.array(CLASSES)
truth = report_predictions["true_index"].to_numpy()
predicted = report_predictions[selection["winner"]].to_numpy()
wrong = truth != predicted

confusions = (pd.Series([f"{class_names[a]} -> {class_names[b]}"
                         for a, b in zip(truth[wrong], predicted[wrong])])
              .value_counts().head(12))

print(f"{wrong.sum():,} errors out of {len(truth):,} ({wrong.mean() * 100:.1f}%)")
fig, ax = plt.subplots(figsize=(8, 4))
confusions.sort_values().plot.barh(ax=ax, color=PALETTE[1])
ax.set(title=f"12 most frequent confusions, {selection['winner']}",
       xlabel="Reporting-split errors")
plt.tight_layout()
save_figure("fig05_top_confusions")
plt.show()

cost = result_table[result_table["model"].isin(ARMS)].copy()
cost["size (MB)"] = cost["model_bytes"] / 1e6
cost["fit (min)"] = cost["train_seconds"] / 60
cost["inference (ms/image)"] = cost["prediction_seconds"] / cost["rows"] * 1000
# Where the work happens is the deployment question, not a detail: the HOG descriptor is a
# CPU routine and the SVM is a matrix multiply, so the winner needs no accelerator at all.
cost["device"] = ["CPU" if ARM_FAMILY[arm] == "hog_svm" else DEVICE.type.upper()
                  for arm in cost["model"]]

display(cost[["model", "macro_f1", "size (MB)", "fit (min)",
              "inference (ms/image)", "device"]].round(3).set_index("model"))

_winner = cost.loc[cost["model"] == selection["winner"]].iloc[0]
print(f"Selected model: {_winner['size (MB)']:.1f} MB, "
      f"{_winner['inference (ms/image)']:.2f} ms/image on {_winner['device']}")


set_phase("test_predictions")
template = pd.read_csv(PREDICTION_TEMPLATE, dtype={"id": str})
assert template["id"].is_unique and TARGET in template.columns

test_frame = pd.DataFrame({"path": [str(TEST_IMAGE_DIR / f"{name}.jpg")
                                    for name in template["id"]]})
missing = [path for path in test_frame["path"] if not Path(path).exists()]
assert not missing, f"{len(missing)} test images named by the template are missing"

test_images = build_image_cache(test_frame, "test")
model = load_final(selection["winner"])
test_pred = predict_images(selection["winner"], model, test_images)

predictions = template.copy()
predictions[TARGET] = class_names[test_pred]
assert predictions["id"].equals(template["id"]), "Row order changed against the template"
assert predictions[TARGET].notna().all()

predictions.to_csv(PREDICTION_DIR / "task1_predictions.csv", index=False)
release(model, test_images)

print(f"{len(predictions):,} predictions covering "
      f"{predictions[TARGET].nunique()} distinct article types")
display(predictions.head())
display(predictions[TARGET].value_counts().head(8).rename("predicted"))

deployment = dict(
    arm=selection["winner"],
    family=selection["winner_family"],
    regime=selection["recipes"][selection["winner"]].get("regime"),
    svm_offset_tau=(selection["svm_offset_tau"]
                    if selection["winner"] == "hog_svm_offset" else None),
    artifact=FINAL_PATHS[selection["winner"]].name,
    classes=CLASSES,
    image_target_size=list(IMAGE_TARGET_SIZE),
    recipe=selection["recipes"][selection["winner"]],
    normalisation_mean=NORM_MEAN.tolist(),
    normalisation_std=NORM_STD.tolist(),
    hog_params={k: list(v) if isinstance(v, tuple) else v for k, v in HOG_PARAMS.items()}
    if ARM_FAMILY[selection["winner"]] == "hog_svm" else None,
    decision=selection["rule"],
    svm_outputs="class decisions, not calibrated probabilities",
    reporting_scores={row["model"]: row["macro_f1"] for row in results},
    source_notebook="01_task1_article_type_classification.ipynb",
)
(OUTPUT_DIR / "deployment.json").write_text(json.dumps(deployment, indent=2), encoding="utf-8")

# The protocol this run followed, recorded so a result can be traced back to the settings
# and the exact rows that produced it. Two identities are worth keeping apart: the protocol
# says what was done, the input digest says what it was done to.
protocol = dict(
    seed=RANDOM_STATE, quick_run=QUICK_RUN, target=TARGET,
    parallelism="DDP" if WORLD_SIZE > 1 else "single", world_size=WORLD_SIZE,
    batchnorm="synchronized" if WORLD_SIZE > 1 else "local",
    validation="rank-strided rows; global ordered logits",
    cpu_policy="all available cores in CPU phases; one native thread per HOG worker",
    amp_enabled=AMP_ENABLED, amp_dtype=str(AMP_DTYPE) if AMP_ENABLED else "float32",
    training_drop_last=True, final_batch_padding="undersized datasets only: repeat to world_size",
    image_target_size=list(IMAGE_TARGET_SIZE), batch_size=BATCH_SIZE,
    reporting_share=REPORTING_SHARE, tuning_share=TUNING_SHARE,
    search_epochs=SEARCH_EPOCHS, confirm_epochs=CONFIRM_EPOCHS,
    patience=PATIENCE, warmup_epochs=WARMUP_EPOCHS, label_smoothing=LABEL_SMOOTHING,
    augmentation=dict(flip=AUG_FLIP_PROBABILITY, rotation=AUG_ROTATION_DEGREES,
                      translate=AUG_TRANSLATE_FRACTION, jitter=AUG_JITTER_STRENGTH),
    search_protocol="independent architecture/regime grids; pilot-refined LR range",
    search_trials=24, confirmation_seeds=[RANDOM_STATE],
    neural_grid=NEURAL_GRID, svm_grid=SVM_GRID, families=list(FAMILIES),
    arms=list(ARMS), imbalance_regimes=list(IMBALANCE_REGIMES),
    resample_power=RESAMPLE_POWER, svm_offset_tau=SVM_OFFSET_TAU,
    imbalance="class-balanced cross-entropy, n / (K * count)",
    versions=dict(python=platform.python_version(), torch=torch.__version__,
                  numpy=np.__version__, pandas=pd.__version__,
                  sklearn=__import__("sklearn").__version__),
)

# The manifest hashes the manifest file and the split membership rather than every image
# byte: it is the same guarantee at a fraction of the cost, because the manifest already
# names each file and the split lists fix which rows went where.
def digest(*parts):
    running = hashlib.sha256()
    for part in parts:
        running.update(json.dumps(part, sort_keys=True, default=str).encode())
    return running.hexdigest()


inputs = dict(
    manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
    rows=dict(fit=len(fit_frame), tuning=len(tune_frame), reporting=len(report_frame)),
    n_classes=N_CLASSES,
    split_sha256=digest(sorted(fit_frame["id"].astype(str)),
                        sorted(tune_frame["id"].astype(str)),
                        sorted(report_frame["id"].astype(str))),
)

# A manifest of everything this run wrote, with a digest of each file. "All the outputs were
# produced" is only a checkable claim if something records what was written.
manifest = {"selected_family": selection["winner"], "selection": selection,
            "protocol": protocol, "inputs": inputs, "files": {}}
for path in sorted(OUTPUT_DIR.rglob("*")):
    if path.is_file() and path.name != "run.json":
        manifest["files"][path.relative_to(OUTPUT_DIR).as_posix()] = dict(
            bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(OUTPUT_DIR / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print(f"Selected: {deployment['family']} ({deployment['artifact']})")
print(f"Input digest: {inputs['split_sha256'][:12]} over {sum(inputs['rows'].values()):,} rows")
print(f"Manifest records {len(manifest['files'])} files under "
      f"{OUTPUT_DIR.relative_to(REPO_ROOT)}")
for name in sorted(manifest["files"]):
    print("   ", name)

RUN_SOTA = os.environ.get('TASK1_RUN_SOTA', '1') == '1'                 # False skips Section 7 entirely
RUN_SOTA_FINETUNE = True        # False keeps the probe (7.4) and skips fine-tuning (7.5)

SOTA_BATCH_SIZE = 64            # embedding extraction; raise it on a larger card
SOTA_FINETUNE_BATCH = 32
SOTA_FINETUNE_EPOCHS = 1 if QUICK_RUN else 5
SOTA_FINETUNE_LR = 3e-5         # a pretrained backbone is adjusted, not trained: two
                                # orders below the 3e-4 the from-scratch models use
SOTA_PROBE_MAX_ITER = 200 if QUICK_RUN else 2000

SOTA_DIR = OUTPUT_DIR / "sota"
(SOTA_DIR / "embeddings").mkdir(parents=True, exist_ok=True)

# The weights are gated for DINOv3. The token is read from the environment and is never
# written into the notebook; without one, 7.2 falls back and says so.
HF_TOKEN = os.environ.get("HF_TOKEN") or None

print(f"RUN_SOTA={RUN_SOTA}  finetune={RUN_SOTA_FINETUNE}  "
      f"epochs={SOTA_FINETUNE_EPOCHS}  token={'set' if HF_TOKEN else 'not set'}")
if RUN_SOTA:
    import importlib.util, subprocess, sys
    if importlib.util.find_spec("transformers") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers>=4.56,<6", "sentencepiece"])


SOTA_SPECS = [('siglip2', 'google/siglip2-base-patch16-224', None, 'image-text contrastive'), ('convnext', 'facebook/convnext-base-224-22k', 'facebook/convnext-tiny-224', 'supervised in21k, convolutional')]

BACKBONES = {}

if RUN_SOTA:
    from transformers import AutoImageProcessor, AutoModel

    def load_backbone(spec):
        """Load a backbone, falling back when the primary is gated or unavailable.

        Returns:
            (key, processor, model, resolved_checkpoint, note), or None if nothing loaded.
        """
        key, primary, fallback, note = spec
        for candidate in (primary, fallback):
            if candidate is None:
                continue
            try:
                processor = AutoImageProcessor.from_pretrained(candidate, token=HF_TOKEN)
                model = AutoModel.from_pretrained(candidate, token=HF_TOKEN)
                # SigLIP loads as a dual encoder whose forward requires input_ids, and
                # there is no text here. Its vision tower is the part being compared, so
                # it is unwrapped; the single-tower backbones have no such attribute and
                # pass through untouched.
                model = getattr(model, "vision_model", model)
                return key, processor, model.to(DEVICE).eval(), candidate, note
            except Exception as error:            # noqa: BLE001 - any failure falls back
                print(f"  [{key}] {candidate}: {type(error).__name__}: {str(error)[:110]}")
        return None

    for spec in SOTA_SPECS:
        print(f"loading {spec[0]} ...")
        loaded = load_backbone(spec)
        if loaded is None:
            print(f"  [{spec[0]}] unavailable; skipped")
            continue
        key, processor, model, resolved, note = loaded
        BACKBONES[key] = dict(processor=processor, model=model, checkpoint=resolved,
                              note=note, requested=spec[1], is_fallback=resolved != spec[1],
                              params=sum(p.numel() for p in model.parameters()))
        flag = "  (FALLBACK)" if BACKBONES[key]["is_fallback"] else ""
        print(f"  [{key}] {resolved}  {BACKBONES[key]['params'] / 1e6:.0f}M params{flag}")

    assert BACKBONES, "No backbone loaded; set RUN_SOTA = False or fix network access."
    print(f"\n{len(BACKBONES)} of {len(SOTA_SPECS)} backbones available")
else:
    print("RUN_SOTA is False; Section 7 skipped")

def sota_image_cache(frame, description, existing):
    """Reuse a cache Section 5 already built; rebuild only if it was released."""
    if existing is not None and len(existing) == len(frame):
        print(f"{description}: reusing the Section 5 cache ({len(frame):,} images)")
        return existing
    return build_image_cache(frame, description)


if RUN_SOTA:
    sota_train_frame = refit_frame
    sota_train_images = sota_image_cache(refit_frame, "refit",
                                         globals().get("refit_images"))
    sota_report_images = sota_image_cache(report_frame, "reporting",
                                          globals().get("report_images"))
    y_sota_train = to_indices(sota_train_frame)

    # Section 5 asserts its own splits are disjoint. It is asserted again here because this
    # section recombines fit and tuning into one training population, and a recombination
    # is exactly where a leak would enter unnoticed.
    _overlap = set(sota_train_frame["id"].astype(str)) & set(report_frame["id"].astype(str))
    assert not _overlap, f"Leakage: {len(_overlap)} ids in both training and reporting"
    assert np.array_equal(y_sota_train, y_refit), "training labels drifted from Section 5"

    print(f"training rows  : {len(sota_train_frame):,}  (the Section 5.1 refit rows)")
    print(f"reporting rows : {len(report_frame):,}  (the Section 5.3 rows)")
    print(f"classes        : {N_CLASSES}   native image size: "
          f"{sota_train_images.shape[2]}x{sota_train_images.shape[1]}")

@torch.no_grad()
def sota_embed(key, images, description):
    """Pooled features for an image cache, in row order, cached to disk.

    Extraction is the expensive step; re-running the notebook to adjust the probe should
    not pay for it a second time.
    """
    set_phase("pretrained:embed:" + key + ":" + description)
    entry = BACKBONES[key]
    # The resolved checkpoint is part of the filename, not just the family name. Otherwise a
    # run where the DINOv3 gate opened would leave dinov3_train.npy behind, and a later run
    # that fell back to DINOv2 would read it and report DINOv2's row from DINOv3's features.
    slug = entry["checkpoint"].replace("/", "_")
    cache_path = SOTA_DIR / "embeddings" / f"{key}_{slug}_{description}.npy"
    if cache_path.exists():
        features = np.load(cache_path)
        if len(features) == len(images):
            print(f"  [{key}] {description}: {features.shape} from cache")
            return features

    start, chunks = time.time(), []
    for begin in range(0, len(images), SOTA_BATCH_SIZE):
        batch = list(images[begin:begin + SOTA_BATCH_SIZE])
        inputs = entry["processor"](images=batch, return_tensors="pt").to(DEVICE)
        outputs = entry["model"](**inputs)
        # pooler_output where the architecture defines one, otherwise mean-pool the token
        # sequence -- mean rather than the CLS token, because not every backbone here was
        # trained to make CLS a summary of the image.
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state.mean(dim=1)
        chunks.append(pooled.float().flatten(1).cpu().numpy())

    features = np.concatenate(chunks, axis=0)
    np.save(cache_path, features)
    print(f"  [{key}] {description}: {features.shape} in {time.time() - start:.0f}s")
    return features


sota_rows, sota_predictions = [], {}

if RUN_SOTA:
    from sklearn.linear_model import LogisticRegression

    for key, entry in BACKBONES.items():
        print(f"\n=== probe {key} ({entry['checkpoint']}) ===")
        x_train = sota_embed(key, sota_train_images, "train")
        x_report = sota_embed(key, sota_report_images, "reporting")

        start = time.time()
        probe = LogisticRegression(max_iter=SOTA_PROBE_MAX_ITER, class_weight="balanced",
                                   n_jobs=-1, random_state=RANDOM_STATE)
        probe.fit(x_train, y_sota_train)
        predicted = probe.predict(x_report)
        name = f"{key}_probe"
        sota_predictions[name] = predicted

        # score_row is Section 5.3's own function: same metric, same buckets, same rows.
        row = score_row(name, predicted, prediction_seconds=float("nan"),
                        train_seconds=time.time() - start, model_bytes=0)
        row.update(protocol="linear probe", backbone=entry["checkpoint"],
                   pretrained=True, submittable=False, is_fallback=entry["is_fallback"],
                   params=entry["params"], feature_dim=int(x_train.shape[1]))
        sota_rows.append(row)
        print(f"  macro-F1 {row['macro_f1']:.4f}  accuracy {row['accuracy']:.4f}  "
              f"({row['train_seconds']:.0f}s to fit the probe)")

def sota_finetune(key, epochs=SOTA_FINETUNE_EPOCHS):
    """Fine-tune one backbone end to end and score it on the reporting rows."""
    from transformers import AutoModelForImageClassification

    set_phase("pretrained:finetune:" + key)
    entry = BACKBONES[key]
    model = AutoModelForImageClassification.from_pretrained(
        entry["checkpoint"], num_labels=N_CLASSES, ignore_mismatched_sizes=True,
        token=HF_TOKEN).to(DEVICE)

    def batches(images, labels, shuffle):
        order = (torch.randperm(len(images)).numpy() if shuffle
                 else np.arange(len(images)))
        for begin in range(0, len(order), SOTA_FINETUNE_BATCH):
            index = order[begin:begin + SOTA_FINETUNE_BATCH]
            pixel = entry["processor"](images=list(images[index]),
                                       return_tensors="pt")["pixel_values"]
            yield pixel.to(DEVICE), torch.as_tensor(labels[index], device=DEVICE)

    # The same inverse-frequency weighting the reweight arms use, on the same rows.
    counts = np.bincount(y_sota_train, minlength=N_CLASSES)
    weights = torch.as_tensor(len(y_sota_train) / (N_CLASSES * np.maximum(counts, 1)),
                              dtype=torch.float32, device=DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=SOTA_FINETUNE_LR,
                                  weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    scaler = make_scaler()          # disabled under bf16, which needs no scaling

    start = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_start, total, seen = time.time(), 0.0, 0
        for pixel, targets in batches(sota_train_images, y_sota_train, shuffle=True):
            with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=AMP_ENABLED):
                loss = F.cross_entropy(model(pixel_values=pixel).logits, targets,
                                       weight=weights)
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            total += loss.item() * len(targets)
            seen += len(targets)
        schedule.step()
        print(f"  [{key} finetune] epoch {epoch + 1}/{epochs} loss {total / seen:.4f} "
              f"({time.time() - epoch_start:.0f}s)")
    train_seconds = time.time() - start

    model.eval()
    predicted, predict_start = [], time.time()
    with torch.no_grad():
        for pixel, _ in batches(sota_report_images, y_report, shuffle=False):
            with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=AMP_ENABLED):
                predicted.append(model(pixel_values=pixel).logits.argmax(1).cpu().numpy())
    predicted = np.concatenate(predicted)

    row = score_row(f"{key}_finetune", predicted,
                    prediction_seconds=time.time() - predict_start,
                    train_seconds=train_seconds, model_bytes=0)
    row.update(protocol="fine-tuned", backbone=entry["checkpoint"], pretrained=True,
               submittable=False, is_fallback=entry["is_fallback"],
               params=entry["params"], feature_dim=None)
    release(model)
    return row, predicted

if RUN_SOTA and RUN_SOTA_FINETUNE:
    for key in list(BACKBONES):
        print(f"\n=== fine-tune {key} ===")
        try:
            row, predicted = sota_finetune(key)
        except torch.cuda.OutOfMemoryError:
            print(f"  [{key}] out of memory at batch {SOTA_FINETUNE_BATCH}; skipped. "
                  f"Lower SOTA_FINETUNE_BATCH and re-run this cell.")
            torch.cuda.empty_cache()
            continue
        sota_rows.append(row)
        sota_predictions[row["model"]] = predicted
        print(f"  macro-F1 {row['macro_f1']:.4f}  accuracy {row['accuracy']:.4f}  "
              f"({row['train_seconds'] / 60:.1f} min)")
elif RUN_SOTA:
    print("RUN_SOTA_FINETUNE is False; the probe results stand alone")

# The backbones are the largest thing in memory by a wide margin and nothing below needs
# them: only the predictions and the rows are still in play. BACKBONES holds the only
# remaining reference, so clearing the entry is what frees the model -- handing it to
# release() while the dict still points at it would free nothing at all. The processors are
# kept, since they are small and name what ran.
if RUN_SOTA:
    for _entry in BACKBONES.values():
        _entry["model"] = None
    release()

if RUN_SOTA:
    assert sota_rows, ("No pretrained result was produced. Section 7.4 has to run before "
                       "this cell, and at least one backbone has to survive 7.2.")
    sota_table = pd.DataFrame(sota_rows)

    scratch = result_table[result_table["model"].isin(ARMS)].copy()
    scratch["protocol"] = "from scratch"
    scratch["pretrained"] = False
    scratch["submittable"] = True
    scratch["backbone"] = None
    scratch["is_fallback"] = False

    bucket_names = [name for name, _, _ in SUPPORT_BUCKETS]
    shared = (["model", "protocol", "backbone", "pretrained", "submittable", "is_fallback",
               "macro_f1", "accuracy", "weighted_f1"] + bucket_names)
    comparison = pd.concat([scratch.reindex(columns=shared),
                            sota_table.reindex(columns=shared)],
                           ignore_index=True).sort_values("macro_f1", ascending=False)

    best_scratch = float(scratch["macro_f1"].max())
    best_pretrained = float(sota_table["macro_f1"].max())
    comparison["vs best submittable"] = (comparison["macro_f1"] - best_scratch).round(4)

    sota_table.to_csv(TABLE_DIR / "sota_results.csv", index=False)
    comparison.to_csv(TABLE_DIR / "sota_comparison.csv", index=False)
    display(comparison.round(4).set_index("model"))

    print(f"best submittable (from scratch): {best_scratch:.4f} macro-F1  "
          f"[{scratch.loc[scratch['macro_f1'].idxmax(), 'model']}]")
    print(f"best pretrained  (reference)   : {best_pretrained:.4f} macro-F1  "
          f"[{sota_table.loc[sota_table['macro_f1'].idxmax(), 'model']}]")
    print(f"gap                            : {best_pretrained - best_scratch:+.4f} "
          f"({(best_pretrained / best_scratch - 1) * 100:+.1f}% relative)")

if RUN_SOTA:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    protocol_colour = {"from scratch": PALETTE[0], "linear probe": PALETTE[1],
                       "fine-tuned": PALETTE[2]}

    ordered = comparison.sort_values("macro_f1")
    axes[0].barh(ordered["model"], ordered["macro_f1"],
                 color=[protocol_colour[p] for p in ordered["protocol"]])
    axes[0].axvline(best_scratch, color="#6b7280", ls=":",
                    label="best submittable")
    axes[0].set(title="Reporting split, macro-F1", xlabel="Macro-F1")
    axes[0].legend(fontsize=8, loc="lower right")

    for _, row in comparison.iterrows():
        axes[1].plot(bucket_names, [row[name] for name in bucket_names], marker="o",
                     alpha=0.85, color=protocol_colour[row["protocol"]],
                     ls="-" if row["submittable"] else "--", label=row["model"])
    axes[1].set(title="Macro-F1 by class support (solid = submittable)",
                ylabel="Macro-F1", xlabel="")
    axes[1].tick_params(axis="x", labelrotation=20)
    axes[1].legend(fontsize=6, ncol=2)

    plt.tight_layout()
    save_figure("fig07_sota_comparison")
    plt.show()

if RUN_SOTA:
    # Which checkpoint each row actually came from, written next to the numbers. A table
    # saying "dinov3" while a fallback was loaded would be the one error in this section
    # that a reader could not catch.
    sota_provenance = dict(
        purpose="pretrained reference comparison; not submittable, not independent data",
        reporting_rows=int(len(report_frame)),
        training_rows=int(len(sota_train_frame)),
        classes=int(N_CLASSES),
        native_image_size=[int(sota_train_images.shape[2]),
                           int(sota_train_images.shape[1])],
        best_submittable_macro_f1=best_scratch,
        best_pretrained_macro_f1=best_pretrained,
        finetune_ran=bool(RUN_SOTA_FINETUNE),
        finetune_epochs=int(SOTA_FINETUNE_EPOCHS),
        backbones={key: dict(requested=entry["requested"], resolved=entry["checkpoint"],
                             is_fallback=bool(entry["is_fallback"]),
                             signal=entry["note"], params=int(entry["params"]))
                   for key, entry in BACKBONES.items()},
        skipped=[spec[0] for spec in SOTA_SPECS if spec[0] not in BACKBONES],
    )
    (OUTPUT_DIR / "sota_provenance.json").write_text(
        json.dumps(sota_provenance, indent=2), encoding="utf-8")

    fallbacks = [key for key, entry in BACKBONES.items() if entry["is_fallback"]]
    print("fallback checkpoints used:", ", ".join(fallbacks) if fallbacks else "none")
    print("backbones skipped entirely:",
          ", ".join(sota_provenance["skipped"]) or "none")

set_phase('complete')
if WORLD_SIZE == 1:
    _telemetry.stop()
    archive_path = package_run(RUN_ROOT, WORK_ROOT, QUICK_RUN)
else:
    print('The parent will close telemetry and package the outputs.')
