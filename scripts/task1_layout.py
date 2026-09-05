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
    # The seed study is the longest job even with the seed-42 checkpoints copied in, and its
    # runs are independent, so it splits by seed. These two take one seed each; seed 42 is not
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
    # The seed figures are the recorded run's own per-seed lines, which is why they differ so
    # much: seed 2024's backbone took 1850 s against seed 1337's 1249 s for the same work.
    "seeds": "~64 min",            # 1587 s (seed 1337) + 2268 s (seed 2024), with the seed-42
                                   # checkpoints copied in. Without them the job also retrains
                                   # seed 42, which is the other ~49 min of the ~113 min the
                                   # recorded run spent here. See PREREQUISITES.
    "seeds_1337": "~27 min",       # 226 s CNN + 1249 s ResNet + 112 s stage-2 head
    "seeds_2024": "~38 min",       # 227 s CNN + 1850 s ResNet + 191 s stage-2 head
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
    "sweep": (
        "hard",
        {"model_resnet_stage1.pt": "resnet"},
        "This job trains no backbone. It attaches five stage-2 heads to the banked Section 6 "
        "stage-1 ResNet, so without that file it stops on `assert source_backbone is not "
        "None` having done nothing. The alternative is to pair the sweep onto the resnet "
        "machine instead of running it here, as `JOB_FILTER = {\"resnet\", \"sweep\"}`.",
    ),
    "seeds": (
        "soft",
        {"model_cnn.pt": "cnn", "model_resnet_decoupled.pt": "resnet"},
        "These make seed 42 free. The seed study reuses their validation logits rather than "
        "retraining that seed, which is exactly what the combine machine does. Without them "
        "this job trains seed 42 under its own keys -- about 49 minutes of measured work that "
        "the combine machine then discards, because it reuses the Section 6 logits for that "
        "row. The run is correct either way; only the time is wasted.",
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

# Only these seven existed at LEGACY_REVISION. sweep, seeds_1337 and seeds_2024 were born
# generated, so there is no hand-derived file for them to be checked against.
LEGACY_JOBS = ["hog_svm", "cnn", "resnet", "logit_adjusted", "seeds", "phase2", "lrsearch"]

# The legacy layout differs from JOBS in where the P2_FINER_MAP note sits, in the two
# trailing cells the workers still carry, and in one anchor: `wanted_seeds` did not exist
# yet, so at that revision the seed cell opened with the plain `wanted` call.
LEGACY_P2_NOTE_OWNER = "lrsearch"
LEGACY_TRAILING = [
    "#### A Note on the Predicted Distribution",
    "## 10. Decision Log, Limitations, and What to Tune Next",
]
LEGACY_ANCHORS = {
    "seeds": ["## 7. Seed Variance", 'if wanted("seeds"):'],
}

# Sections that are a job now but were combine-only then. Without these the sweep section,
# whose job did not exist at LEGACY_REVISION, is unowned in legacy mode and falls through to
# the spine, which would put it in all seven hand-derived workers and fail the gate.
LEGACY_COMBINE_ONLY = [
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
        if legacy:
            if name not in LEGACY_JOBS:
                continue
            anchors = LEGACY_ANCHORS.get(name, anchors)
        for anchor in anchors:
            owners[resolve(cells, anchor)] = name
    if legacy:
        owners[resolve(cells, "#### On `P2_FINER_MAP`")] = LEGACY_P2_NOTE_OWNER

    combine_only = {resolve(cells, anchor) for anchor in COMBINE_ONLY}
    if legacy:
        combine_only |= {resolve(cells, anchor) for anchor in LEGACY_COMBINE_ONLY}
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
