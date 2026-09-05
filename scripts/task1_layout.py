"""Shared description of how the Task 1 worker notebooks derive from the combine notebook.

The parallel run rests on one invariant: every cell a worker shares with the combine
notebook is byte-identical to it. That is what guarantees the machines agree on
RUN_FINGERPRINT, and therefore that a checkpoint trained on one is accepted by another.
This module is the single place that invariant is written down; `check_task1_workers.py`
enforces it and `make_task1_workers.py` produces files that satisfy it by construction.

Cells are addressed by a distinctive substring of their source rather than by index, so
inserting a cell into the combine notebook does not silently reshuffle every worker.

The derivation rule, in full:

    worker = [worker header]                     replaces the combine title cell
           + [spine cells before the worker stop, minus COMBINE_ONLY, minus other jobs]
           + [this job's own cells, in combine order]
           + [worker done cell]                  replaces the combine worker-stop cell

where "spine" is everything not claimed by a job and not declared combine-only. A new
shared cell therefore joins every worker automatically; only a new combine-only cell has
to be declared here.
"""

# --- Anchors -----------------------------------------------------------------------------
# Each anchor must match exactly one cell of the combine notebook. `resolve` asserts that,
# which is what turns a renamed or duplicated cell into a loud failure rather than a worker
# that quietly loses a section.

TITLE = "# Task 1: Fashion Item Type Classification"
# The mode cell is anchored on the line below the assignment: the worker-stop cell quotes
# "JOB_FILTER = None" in its message, so the assignment alone is not unique.
JOB_FILTER_CELL = "COMBINE = JOB_FILTER is None"
JOB_FILTER_LINE = "JOB_FILTER = None"
WORKER_STOP = "# --- Worker stop ---"

# Cells the combine machine keeps to itself: cross-model analysis that reads results no
# single worker has. Everything from WORKER_STOP onward is combine-only by position and
# needs no anchor here.
COMBINE_ONLY = [
    TITLE,
    'ablation = pd.concat([r for r in RESULTS if r["Model"].iloc[0].startswith("3")]',
    "single_plain = predict_probabilities(",
    "### 7.7 Reading Run A: The Localised-Detail Pairs",
    "READOUT_PAIRS = [",
    "# --- Reproduction check ---",
    "progress = pd.concat(RESULTS, ignore_index=True)",
]

# --- Jobs --------------------------------------------------------------------------------
# job name -> (subtitle for the worker header, [anchors of the cells only this job runs])
#
# A job's cells are the ones gated on `wanted("<job>")` plus the markdown that introduces
# them. Everything else is spine.
JOBS = {
    "hog_svm": (
        "Model 1 - HOG + Linear SVM",
        ['if wanted("hog_svm"):'],
    ),
    "cnn": (
        "Model 2 - CNN from scratch",
        ['if wanted("cnn"):'],
    ),
    "resnet": (
        "Model 3 - ResNet + decoupled classifier",
        [
            "### 6.1 Stage 1: Instance-Balanced Training",
            "resnet, resnet_history, resnet_logits = train_or_restore(",
            "decoupled = build_decoupled(resnet)",
        ],
    ),
    "logit_adjusted": (
        "Ablation - logit-adjusted ResNet",
        [
            "### 6.3 Ablation: Which Change Paid?",
            'if wanted("logit_adjusted"):',
        ],
    ),
    "seeds": (
        "Seed variance study (all three seeds)",
        [
            "## 7. Seed Variance",
            "ACTIVE_SEEDS = wanted_seeds(SEEDS)",
        ],
    ),
    # The seed study is the critical path at roughly 113 minutes, more than twice the next
    # longest job, and its runs are independent. These two take one seed each; seed 42 is not
    # a job at all, because the cnn and resnet workers already produce exactly those weights.
    "seeds_1337": ("Seed variance study (seed 1337)", []),
    "seeds_2024": ("Seed variance study (seed 2024)", []),
    "phase2": (
        "Phase 2 - multi-task ResNet",
        [
            'if RUN_PHASE2 and wanted("phase2"):',
            # Phase-2 commentary. The hand-derived workers filed this with lrsearch, which
            # is how it reached the machine that cannot act on it and not the one that can.
            "#### On `P2_FINER_MAP`, and What It Would and Would Not Show",
        ],
    ),
    "sweep": (
        "Stage-2 sampler sweep",
        [
            "### 7.8 Sampler Strength: A Sweep Rather Than a Second Full Model",
            "RUN_SAMPLER_SWEEP = True",
        ],
    ),
    "lrsearch": (
        "Learning-rate search (CNN + ResNet)",
        [
            "## 7.9 Learning-Rate Search",
            "# --- Learning-rate search over the two non-baseline models",
            "if RUN_LR_SEARCH and (COMBINE or wanted(",
        ],
    ),
}

# Wall time per job, measured from the recorded run stored in the combine notebook's outputs
# (the `... this session` lines) rather than estimated. That run was throttled to a 65% duty
# cycle, and the same work varied by up to 2x within it from contention, so these are the
# right order of magnitude and a reliable ranking rather than precise figures.
RUNTIMES = {
    "hog_svm": "~4 min",           # restored in the recorded run; not measured
    "cnn": "~7 min",               # 421 s
    "resnet": "~41 min",           # 2356 s stage 1, plus stage 2
    "logit_adjusted": "~35 min",   # 2122 s
    "phase2": "~27 min",           # 1471 s stage 1, plus stage 2
    "sweep": "~8 min",             # 5 heads x 10 epochs, frozen backbone
    "lrsearch": "~56 min",         # never run; 6 arms x 15 epochs at the measured epoch cost
    "seeds": "~113 min",           # 3 seeds x (CNN + ResNet + head); see the split below
}

# The layout the hand-derived notebooks actually have, used once to prove the generator
# reproduces them before it is allowed to change anything. It differs from JOBS only in
# where the P2_FINER_MAP note sits, and in the two trailing cells the workers still carry.
LEGACY_P2_NOTE_OWNER = "lrsearch"
LEGACY_TRAILING = [
    "#### A Note on the Predicted Distribution",
    "## 10. Decision Log, Limitations, and What to Tune Next",
]

# --- Fingerprint regression --------------------------------------------------------------
# RUN_FINGERPRINT is a hash of the problem and the recipe. If it moves, every checkpoint on
# disk is refused and the parallel run has to start over, so it is pinned here.
#
# The data-derived inputs are the ones the manifest fixes; they are taken from the recorded
# run whose outputs are stored in the combine notebook. The normalisation constants are
# float32 values rounded to five decimals, so they carry the float32 bit pattern rather than
# a clean decimal, and are reproduced here exactly as `NORM_MEAN.round(5).tolist()` emits them.
EXPECTED_FINGERPRINT = "e6b15f5c51de"

RECORDED_RUN = {
    "classes": 124,
    "train_rows": 30278,
    "val_rows": 7568,
    "norm": [
        [0.8492100238800049, 0.8325200080871582, 0.8267199993133545],
        [0.2718600034713745, 0.283160001039505, 0.28692999482154846],
    ],
}

# Notebook constant -> key in the fingerprint payload. Anything listed here is read out of
# the notebook source, so editing one of these values in the notebook fails the check.
FINGERPRINT_SCALARS = {
    "TARGET": "target",
    "RANDOM_STATE": "seed",
    "QUICK_RUN": "quick",
    "VALIDATION_SHARE": "validation_share",
    "BATCH_SIZE": "batch",
    "EPOCHS": "epochs",
    "PATIENCE": "patience",
    "LEARNING_RATE": "lr",
    "WEIGHT_DECAY": "weight_decay",
    "WARMUP_EPOCHS": "warmup",
    "LABEL_SMOOTHING": "label_smoothing",
    "LOGIT_ADJUST_TAU": "tau",
}

FINGERPRINT_AUG = [
    "AUG_FLIP_PROBABILITY",
    "AUG_ROTATION_DEGREES",
    "AUG_TRANSLATE_FRACTION",
    "AUG_JITTER_STRENGTH",
]

FINGERPRINT_STAGE2 = ["STAGE2_EPOCHS", "STAGE2_LR", "STAGE2_USE_DROPOUT"]


# Jobs that run another job's cells and differ only in what JOB_FILTER makes those cells do.
CELL_ALIASES = {"seeds_1337": "seeds", "seeds_2024": "seeds"}


def worker_cells(cells, job, legacy=False):
    """Combine-cell indices a worker for `job` holds, in combine order.

    Excludes the three cells a worker owns rather than shares (its header, its mode cell and
    its done cell); those are substituted by the generator and cannot be compared against the
    combine notebook. `legacy` reproduces the hand-derived layout instead of the target one.
    """
    stop = resolve(cells, WORKER_STOP)
    owners = {}
    for name, (_, anchors) in JOBS.items():
        for anchor in anchors:
            owners[resolve(cells, anchor)] = name
    if legacy:
        owners[resolve(cells, "#### On `P2_FINER_MAP`")] = LEGACY_P2_NOTE_OWNER

    combine_only = {resolve(cells, anchor) for anchor in COMBINE_ONLY}
    trailing = {resolve(cells, anchor) for anchor in LEGACY_TRAILING}
    substituted = {stop, resolve(cells, JOB_FILTER_CELL)}

    kept = []
    for index in range(len(cells)):
        # Everything from the worker stop onward is the combine machine's analysis, except
        # the two trailing cells the hand-derived workers still carry.
        if index >= stop and not (legacy and index in trailing):
            continue
        if index in combine_only or index in substituted:
            continue
        if index in owners and owners[index] != CELL_ALIASES.get(job, job):
            continue
        kept.append(index)
    return kept


def resolve(cells, anchor):
    """Index of the one cell whose source contains `anchor`.

    Raises rather than guessing: an anchor that matches nothing or matches twice means the
    notebook moved underneath this file, and continuing would silently drop a section from
    every worker.
    """
    hits = [i for i, (_, source) in enumerate(cells) if anchor in source]
    if len(hits) != 1:
        raise LookupError(
            f"anchor {anchor!r} matched {len(hits)} cells (expected exactly 1)"
            + (f" at indices {hits}" if hits else "")
        )
    return hits[0]
