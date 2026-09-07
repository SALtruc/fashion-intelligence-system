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
    'evaluate_predictions(y_val, plain.argmax(axis=1), plain,',
    "intervals, BOOTSTRAP_DRAWS = bootstrap_macro_f1(contenders)",
    "# --- Reproduction check ---",
    "progress = pd.concat(RESULTS, ignore_index=True)",
]

# COMBINE_ONLY anchors are resolved against whichever notebook is being read, and the legacy
# gate reads the one at LEGACY_REVISION. Where a combine-only cell has since been rewritten,
# the anchor that finds it *there* is recorded here rather than kept in the list above, which
# has to keep finding it in the notebook as it stands now. A cell that did not exist at all at
# that revision -- the Section 7.2 bootstrap -- is dropped from the legacy list instead.
LEGACY_RENAMED = {
    'evaluate_predictions(y_val, plain.argmax(axis=1), plain,':
        "single_plain = predict_probabilities(",
}
LEGACY_ABSENT = ["intervals, BOOTSTRAP_DRAWS = bootstrap_macro_f1(contenders)"]

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
    "sweep": (
        "Stage-2 sampler sweep",
        [
            "### 7.3 Sampler Strength: A Sweep Rather Than a Second Full Model",
            "RUN_SAMPLER_SWEEP = True",
        ],
    ),
    "hogsearch": (
        "HOG + SVM regularisation grid",
        ["# --- Grid 1: HOG + linear SVM, regularisation x class weighting"],
    ),
    "cnnsearch": (
        "CNN learning-rate x weight-decay grid",
        ["# --- CNN: learning rate x weight decay"],
    ),
    "stage2grid": (
        "Stage-2 learning-rate x sampler grid",
        ["# --- Grid 4: stage-2 learning rate x sampler strength"],
    ),
    "lrsearch": (
        "Learning-rate search (CNN + ResNet)",
        [
            "### 7.4 Hyper-parameter Search",
            "# --- ResNet: learning rate x weight decay",
        ],
    ),
}

# Wall time per job. These were measured on Apple Silicon (MPS, fp32, no AMP and no
# on-device cache) by a run whose notebooks are no longer kept, so nothing in this
# repository reproduces them and they cannot be re-derived from it. Read them as a
# reliable *ranking* and an order of magnitude, not as CUDA figures -- a mid-range CUDA
# card with AMP is materially faster. Re-measure on the target hardware and update this
# table before scheduling against it.
#
# The three measured per-epoch costs everything below is built from:
#     CNN backbone     14 s/epoch      ResNet backbone   96 s/epoch
#     stage-2 head     28 s/epoch      liblinear SVM    159 s/fit
# The HOG descriptor itself is 3 s for the whole training set and is not worth modelling.
#
# Nothing here is hashed, and Task 1 runs at full machine capacity, so these are the times a
# machine gives when it is not sharing itself with anything else.
RUNTIMES = {
    "hog_svm": "~3 min",           # 3 s descriptor + 159 s liblinear
    "cnn": "~9 min",               # 552 s measured
    "resnet": "~64 min",           # 3580 s stage 1 measured + 10 x 28 s stage-2 head
    "sweep": "~23 min",            # 5 heads x 10 epochs x 28 s, frozen backbone
    "lrsearch": "~115 min",        # 6 arms x 12 epochs x 96 s
    # The four tuning grids, 6 arms each. The two backbone grids run a reduced 12-epoch
    # budget, so their cost is 6 x 12 x the per-epoch time of the architecture under test.
    "hogsearch": "~16 min",        # 6 liblinear fits; the descriptor is computed once
    "cnnsearch": "~17 min",        # 6 arms x 12 epochs x 14 s
    "stage2grid": "~28 min",       # 6 arms x 10 epochs x 28 s, frozen backbone
}


# --- Cross-job prerequisites -------------------------------------------------------------
# Jobs that need another job's checkpoints on the machine before they start, as
# job -> (severity, {checkpoint filename: producing job}, why).
#
# "hard": the run stops on an assertion, so the machine is idle until the file arrives.
# "soft": the run succeeds and spends the stated time producing checkpoints the combine
#         machine will never read.
#
# This is rendered into the worker header by make_task1_workers, because the header is the
# only thing the operator of that machine reads before starting an hour of work. Stating it
# in a comment inside the cell that needs the file is too late.
PREREQUISITES = {
    "stage2grid": (
        "hard",
        {"model_resnet_stage1.pt": "resnet"},
        "This job trains no backbone. Each of its six arms attaches a fresh stage-2 head to the "
        "banked Section 6 stage-1 ResNet, so without that file it stops on an assertion having "
        "done nothing. Pair it onto the resnet machine as JOB_FILTER = {\"resnet\", "
        "\"stage2grid\"} if copying the checkpoint in first is inconvenient.",
    ),
    "sweep": (
        "hard",
        {"model_resnet_stage1.pt": "resnet"},
        "This job trains no backbone. It attaches five stage-2 heads to the banked Section 6 "
        "stage-1 ResNet, so without that file it stops on `assert source_backbone is not "
        "None` having done nothing. The alternative is to pair the sweep onto the resnet "
        "machine instead of running it here, as `JOB_FILTER = {\"resnet\", \"sweep\"}`.",
    ),
}

# --- The legacy gate ---------------------------------------------------------------------
# The hand-derived notebooks are what the generator had to reproduce before it was allowed
# to change anything, and reproducing them is the only evidence that `worker_cells` encodes
# the real cell-selection rule rather than a plausible-looking guess.
#
# Both sides of that comparison are read from LEGACY_REVISION, never from the working tree.
# Reading the workers from disk (which is what this did originally) stops proving anything
# the moment the generator rewrites them: it then compares the generator against its own
# output. Reading the combine notebook from disk is just as wrong, because the hand-derived
# workers were cut from the combine notebook as it stood at that commit, not as it stands
# now. Pinning both keeps the gate meaningful for as long as the history exists.
# Spelled in full: an abbreviation is only unique until the history grows into it.
LEGACY_REVISION = "cd44bc8f16c5e9676edbd57b40ac86887fbb6a5d"

# Only these existed at LEGACY_REVISION as hand-derived files. sweep, hogsearch, cnnsearch and
# stage2grid were born generated, so there is nothing to check them against; `seeds` had one
# and is no longer a job, so its cells are listed under LEGACY_COMBINE_ONLY instead and the
# gate now proves the rule on four workers rather than five.
LEGACY_JOBS = ["hog_svm", "cnn", "resnet", "lrsearch"]

# The legacy layout differs from JOBS in the two trailing cells the workers still carry, and in
# one job's anchors: the hyper-parameter search was a learning-rate search over both
# architectures in one cell rather than the four separate grids it is now.
LEGACY_TRAILING = [
    "#### A Note on the Predicted Distribution",
    "## 10. Decision Log, Limitations, and What to Tune Next",
]
LEGACY_ANCHORS = {
    "lrsearch": [
        "## 7.9 Learning-Rate Search",
        "# --- Learning-rate search over the two non-baseline models",
        'if RUN_LR_SEARCH and (COMBINE or wanted("lrsearch")):',
    ],
}

# The hand-derived worker_lrsearch carried the P2_FINER_MAP note, which sits immediately above
# Section 7.9 and was cut along with it. Naming the owner here rather than leaving the note
# unowned is what keeps it out of the other four workers.
LEGACY_P2_NOTE_OWNER = "lrsearch"

# Sections that were combine-only at LEGACY_REVISION and are a job, or gone, now. Without
# these they are unowned in legacy mode and fall through to the spine, which would put them
# in all five hand-derived workers and fail the gate.
#
# The 6.3 ablation, the whole of Phase 2 and the seed-variance study have since been removed
# from the notebook, so they exist only here; the sampler sweep became the `sweep` job.
LEGACY_COMBINE_ONLY = [
    "## 7. Seed Variance",
    'if wanted("seeds"):',
    "### 6.3 Ablation: Which Change Paid?",
    'if wanted("logit_adjusted"):',
    'if RUN_PHASE2 and wanted("phase2"):',
    "### 7.7 Reading Run A: The Localised-Detail Pairs",
    "# --- Run A readout: the pairs the hypothesis is about",
    "### 7.8 Sampler Strength: A Sweep Rather Than a Second Full Model",
    "# --- Stage-2 sampler sweep on the banked Section 6 backbone",
]

# --- Fingerprint regression --------------------------------------------------------------
# RUN_FINGERPRINT is a hash of the problem and the recipe. If it moves, every checkpoint on
# disk is refused and the parallel run has to start over, so it is pinned here.
#
# The data-derived inputs are the ones the manifest fixes; they are taken from the recorded
# run whose outputs are stored in the combine notebook. Their names are also the payload keys
# that no notebook constant supplies, which is how the checker knows the expected key set is
# complete. The normalisation constants are float32 values rounded to five decimals, so they
# carry the float32 bit pattern rather than a clean decimal, and are reproduced here exactly
# as `NORM_MEAN.round(5).tolist()` emits them.
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

# The assignment that computes RUN_FINGERPRINT, so the checker can parse the payload's dict
# literal instead of trusting the three lists below to still describe it. Reconstructing the
# payload from these lists alone cannot see a key added to or removed from the notebook: the
# real fingerprint moves, the reconstruction does not, and the check reports "unchanged"
# while every checkpoint on disk is being refused. That is precisely the failure it exists
# to catch, so the key set is compared against the notebook rather than assumed.
FINGERPRINT_ASSIGNMENT = "RUN_FINGERPRINT = hashlib.sha1("

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
# Empty since the seed jobs were retired; kept because the mechanism is what lets one set of
# cells be shared by several JOB_FILTER values without duplicating anchors.
CELL_ALIASES = {}

WORKER_JOBS = tuple(JOBS)


def worker_cells(cells, job, legacy=False):
    """Combine-cell indices a worker for `job` holds, in combine order.

    Excludes the three cells a worker owns rather than shares (its header, its mode cell and
    its done cell); those are substituted by the generator and cannot be compared against the
    combine notebook. `legacy` reproduces the hand-derived layout instead of the target one.
    """
    stop = resolve(cells, WORKER_STOP)
    owners = {}
    for name, (_, anchors) in JOBS.items():
        if legacy:
            if name not in LEGACY_JOBS:
                continue
            anchors = LEGACY_ANCHORS.get(name, anchors)
        for anchor in anchors:
            owners[resolve(cells, anchor)] = name
    if legacy:
        owners[resolve(cells, "#### On `P2_FINER_MAP`")] = LEGACY_P2_NOTE_OWNER

    anchors = list(COMBINE_ONLY)
    if legacy:
        anchors = [LEGACY_RENAMED.get(anchor, anchor) for anchor in anchors
                   if anchor not in LEGACY_ABSENT] + list(LEGACY_COMBINE_ONLY)
    combine_only = {resolve(cells, anchor) for anchor in anchors}
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
