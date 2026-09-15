<h1 align="center">👗 Fashion Intelligence System</h1>

<p align="center">
  <b>Four computer-vision models and a visual search engine that read a fashion-catalogue
  photograph and answer: what the item is, which season it is for, who it is for,<br>
  what occasion it suits — and which items look most like it.</b>
</p>

<p align="center">
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-CUDA%2012.8-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="uv" src="https://img.shields.io/badge/env-uv%20locked-261230">
  <img alt="Optuna" src="https://img.shields.io/badge/HPO-Optuna-1F77B4">
  <img alt="FAISS" src="https://img.shields.io/badge/retrieval-FAISS-0467DF">
  <img alt="Models" src="https://img.shields.io/badge/models-trained%20from%20scratch-success">
</p>

<p align="center">
  <a href="#-what-it-does">What it does</a> ·
  <a href="#-results">Results</a> ·
  <a href="#-engineering-highlights">Engineering</a> ·
  <a href="#-quickstart">Quickstart</a> ·
  <a href="#-live-demo">Demo</a>
</p>

---

## 📌 Overview

A **38,000-image fashion catalogue** goes in. Out comes a structured record of every garment —
its article type, its season, its intended wearer and occasion — plus a ranked list of the items
that look most like it.

The interesting part is not that it classifies images. It is **how heavily the catalogue is
long-tailed**: 124 article-type classes where a handful hold thousands of images and dozens hold
fewer than twenty. Every design decision in this repository is an answer to that fact — the choice
of metric, the resampling scheme, the loss weighting, the way models are compared, and the two
techniques that were **tried and then rejected on their own evidence**.

Five contributors · **234 commits over five weeks** · every shipped model **trained from scratch**;
pretrained networks appear only as clearly-labelled reference points, never as a shipped model.

> Built as the Assignment 2 project for **COSC2753 Machine Learning**, RMIT University (2026B).

---

## 🚀 What it does

| | Capability | Question answered | Output space |
|---|---|---|---|
| 👕 | **Article-type classification** | *What kind of item is this?* | 124 classes, `Tshirts` → `Lip Gloss` |
| 🍂 | **Season classification** | *Which season is it for?* | Spring / Summer / Fall / Winter |
| 🎯 | **Gender & usage classification** | *Who is it for, and for what occasion?* | 5 × 8, two heads on one shared body |
| 🔍 | **Visual search** | *What looks like this?* | Top-K ranking over a 34k-image gallery |

---

## 📊 Results

The catalogue is long-tailed, so **macro-F1 is the primary metric everywhere** — accuracy alone
rewards a model for ignoring rare classes. Each task is scored **once**, on a held-out reporting
split it never saw during tuning.

| Task | Shipped model | Primary metric | Runner-up | Baseline floor |
|---|---|---|---|---|
| **1 · Article type** | SmallResNet + resampling | **macro-F1 0.7654** · acc 0.8735 | CNN + resampling 0.7447 | HOG+SVM 0.6345 |
| **2 · Season** | DenseNet-121 | **macro-F1 0.7515** · top-1 0.7561 | EfficientNet-B0 0.7454 | majority class 0.1652 |
| **3 · Gender & usage** | Shared body + 2 heads, class-weighted loss + mirror TTA | **gender 0.7202** · **usage 0.4676** | two separate models | 1-NN 0.534 / 0.327 |
| **4 · Visual search** | ArcFace (ResNet-18) + k-reciprocal & HSV reranking | **mAP@10 0.7755** · P@1 0.7986 | SupCon 0.7274 | conv. autoencoder 0.5024 |

**Where that sits against pretrained models.** A SigLIP2 linear probe reaches macro-F1 `0.8219` on
Task 1 — ahead of our from-scratch `0.7654`, and useful precisely as a scale marker. On Task 4,
FashionCLIP reaches raw mAP@10 `0.6918`, **below** our ArcFace `0.7702`, because ArcFace is trained
directly against the relevance labels the benchmark scores.

> **On Task 3's `usage` score.** `0.4676` is not an under-trained model. Four of the eight `usage`
> classes rest on 79 training images between them, and an oracle handed the *true* `articleType`
> still scores only macro-F1 `0.3872`, with `0.0000` on all four rare classes. The ceiling is in the
> catalogue's labelling, and the work measures it two independent ways rather than asserting it.

---

## 🖼️ Inside the investigation

**The class imbalance every design decision answers to**

![Article-type class distribution](notebooks/task1/figures/fig01_class_distribution.png)

**Task 1 — five models under one protocol, scored once on a held-out reporting split**

![Task 1 reporting comparison](notebooks/task1/figures/fig04_reporting_comparison.png)

**Task 3 — three ways to attach two labels to one convolutional body**

![Multi-task designs](notebooks/task3/figures/01_multitask_designs.svg)

**Task 4 — the controlled retrieval benchmark**

![Task 4 benchmark flow](notebooks/task4/figures/06_benchmark_flow.svg)

---

## 🔬 Engineering highlights

The things worth a second look here are methodological, not architectural.

**🧹 The data was audited, not trusted.** The preprocessing notebook does not take the provided
metadata at face value. It finds **636 duplicate image groups** covering 1,399 images, and **11 test
images byte-identical to a training image** — contamination that silently inflates a test score if
nobody looks — then writes one audited manifest every downstream notebook reads:

| Stage | Rows |
|---|---:|
| Raw metadata | 38,617 |
| Rows with a readable image | 38,612 |
| After collapsing 616 consistent duplicate groups | 37,870 |
| After resolving 23 label conflicts | **37,847** |

**🔒 One frozen split, one shot at the test set.** Hyperparameters and checkpoint epochs are chosen
on a tuning split only. Recipes are then frozen, refit on fit+tuning, and scored **once** on
reporting. Nobody invents their own split — everyone evaluates on the committed files in `splits/`.

**📐 Differences are tested, not eyeballed.** Task 1 reports paired bootstrap intervals on the gaps
between models. The ResNet-vs-CNN interval is `[-0.0020, +0.0393]` — it includes zero, and the
notebook says so instead of declaring a winner.

**❌ Two techniques were rejected on their own evidence.** Externally collected training data showed
**no measurable effect on either Task 3 target** across four runs on two platforms, with the sign
flipping both times. It is reported as a null result rather than quietly dropped. The shipped Task 3
checkpoint is the **median** of three seeds, not the best one.

**🎲 Hyperparameters are searched, not guessed.** Each of Task 4's five retrieval models is tuned
with Optuna against validation mAP@10, then all five are frozen and scored on one common hold-out
gallery — with raw cosine and reranked results reported **separately**, so post-processing is never
mistaken for representation quality.

**♻️ Reproducibility was verified, not assumed.** The repository was checked out into an empty
directory, given only the source data, and run. Tasks 1 and 3 re-predicted from their committed
checkpoints, and the resulting CSV came out **byte-identical** to the submitted one — 204,438 bytes,
SHA-256 `3b6b6930…`. The environment is pinned by `uv.lock` against Python 3.12.0.

**🧪 Information theory before architecture.** Task 3's design choice starts from measurement:
`gender` and `usage` are nearly independent of each other (11.5% / 9.5% mutual-information
reduction) but both strongly tied to `articleType` (66.8% / 43.1%) — which argues for sharing
*features* rather than labels. Three designs were then measured across seeds and platforms to check
that the argument held.

---

## 🧠 What each task investigates

**Task 1 — Article type.** HOG+linear-SVM, a plain CNN and a SmallResNet, each under two imbalance
regimes (re-weighting and resampling), compared on disjoint fit / tuning / reporting splits.

**Task 2 — Season.** Random Forest on engineered features against two fine-tuned CNN families, with
a majority-class floor and an equal-weight selection score across five metrics.

**Task 3 — Gender & usage.** Three multi-task designs, repeated across seeds and platforms, with
class weighting, logit adjustment, TTA, same-dataset auxiliary pretraining and externally collected
data — two of which were rejected on their own evidence.

**Task 4 — Visual search.** A development ladder where each model answers the previous one's
limitation: convolutional autoencoder → triplet margin → supervised contrastive → multi-similarity
→ ArcFace, each tuned with Optuna, then all five frozen and benchmarked together.

---

## 🛠️ Tech stack

| Layer | Tools |
|---|---|
| **Modelling** | PyTorch · torchvision · custom SmallResNet, CNN, ArcFace/SupCon/triplet heads |
| **Classical ML** | scikit-learn (HOG + linear SVM, Random Forest) · scikit-image |
| **Retrieval** | FAISS · k-reciprocal re-ranking · HSV colour reranking |
| **Tuning** | Optuna · iterative-stratification · paired bootstrap significance testing |
| **Reference models** | SigLIP2, FashionCLIP via `transformers` / `timm` — evaluation only |
| **Environment** | Python 3.12.0 · `uv` locked · CUDA 12.8 wheels |
| **Workflow** | Jupyter · pytest gates on the shared library · frozen split files in git |

---

## ⚡ Quickstart

Requires **Python 3.12.0** and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/SALtruc/fashion-intelligence-system.git
cd fashion-intelligence-system
uv sync --frozen
```

Install `uv` first if you do not have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
irm https://astral.sh/uv/install.ps1 | iex
```

**The four trained models are committed** (≈197 MB, one per task plus Task 4's encoded gallery), so
a fresh clone can predict without a GPU or a download link. Each one also runs standalone:

| Task | Checkpoint | Run it alone |
|---|---|---|
| 1 · article type | `artifacts/task1/submitted/` | `python -m src.task1.task1_inference --template <csv> --image-dir <dir> --output <csv>` |
| 2 · season | `artifacts/task2/submitted/` | `python -c "from src.task2_utils import predict_test_set; predict_test_set()"` |
| 3 · gender & usage | `artifacts/task3/submitted/` | `python src/task3/predict_test.py` |
| 4 · visual search | `artifacts/task4/submitted/` | `python src/task4/retrieve_topk.py --images <dir>` |

Task 4 is a retrieval system, so it has no label column — it writes one row per (query, rank) pair.

### Getting the data

**The catalogue is not committed** — it is far past what belongs in a Git repository, and its
licence limits it to this course. With `A2_Fashion.zip` from RMIT Canvas, unpack it as:

```
datasets/
├── train/
│   ├── styles_train.csv
│   └── images_train/            # ~38,600 .jpg
└── test/
    ├── styles_prediction.csv
    └── images_test/             # ~5,800 .jpg
```

That is the only layout you need to get right. Everything resolves the data through
[`src/data_paths.py`](src/data_paths.py), which also accepts the archive's own nested folder, a
Colab mount, and `A2_DATA_ROOT=/somewhere/else`. If something is missing it says so once, naming
every path it looked in, instead of failing several cells later.

```bash
python -c "from src import data_paths; data_paths.check(); print(data_paths.describe())"
```

---

## ▶️ Running the notebooks

Every notebook is saved **with its outputs**, so the full investigation can be read without
executing anything. To re-run, launch from the repository root and pick the project's `.venv` kernel:

```bash
uv run jupyter notebook
```

Run in this order — later notebooks read what earlier ones write:

| Order | Notebook | Produces | Needs |
|---|---|---|---|
| 1 | `notebooks/00_eda_and_preprocessing.ipynb` | `preprocessed_datasets/train_manifest.csv` | the `datasets/` layout · CPU, minutes |
| 2 | `notebooks/task1/01_…` | Task 1 models, tables, figures, predictions | a CUDA GPU |
| 3 | `notebooks/task2/…` | Task 2 checkpoints, comparison, predictions | a CUDA GPU |
| 4 | `notebooks/task3/03_…` | Task 3 checkpoint, `predictions/task3/` | GPU for §2–11; §12's figures need nothing |
| 5 | `notebooks/task4/00…05`, then `06_final_benchmark.ipynb` | five retrieval models, then the hold-out benchmark | a GPU |
| 6 | `notebooks/01_final_prediction.ipynb` | the combined prediction CSV | only the per-task CSVs; trains nothing |

Only step 6 is needed to reproduce the shipped predictions; steps 1–5 retrain the models it loads.

> **Re-running Task 3, or Task 4's `00_preprocessing`, will not reproduce their saved numbers
> exactly.** Both read a shared catalogue table of **37,745** rows that was staged externally; the
> audited manifest here holds **37,847** — the same images plus 102 the earlier table had dropped.
> Nothing is lost, but the splits are redrawn and the metrics move slightly. Task 4's notebooks
> `01`–`06` are unaffected: they read the frozen `splits/task4/` CSVs, not the table.

---

## 📁 Repository structure

```
.
├── notebooks/
│   ├── 00_eda_and_preprocessing.ipynb    # audit, dedup, conflict resolution → train_manifest.csv
│   ├── 01_final_prediction.ipynb         # combines all four tasks into one prediction CSV
│   └── task1/ task2/ task3/ task4/       # one investigation per task, outputs saved
├── src/                                  # shared library: data, models, training, retrieval, metrics
│   ├── data_paths.py                     # the one resolver every task uses to find the data
│   ├── preprocessing.py  task2_utils.py  external_data.py
│   └── task1/  task3/  task4/            # task1 also holds its inference & packaging scripts
├── splits/                               # frozen split files — every task evaluates on these
├── preprocessed_datasets/                # the audited 37,847-row manifest all tasks read
├── artifacts/                            # task*/submitted/ — the four shipped checkpoints
├── predictions/                          # prediction CSVs and per-task result tables
├── data/external_task1/                  # Task 1's committed 60-image independent evaluation set
├── docs/EXTERNAL_DATA.md                 # provenance & licences for externally collected images
├── tests/                                # self-tests for the shared code
└── pyproject.toml · uv.lock              # locked environment
```

---

## 🌐 Live demo

A web interface for trying the classifiers on your own image:

**🔗 [truc-ml-web-dmz9.vercel.app](https://truc-ml-web-dmz9.vercel.app/)**

> The deployed service is a separate application with its own model lineage and **may lag the
> checkpoints selected here**. The notebooks and their saved outputs are the authoritative results
> for every number quoted above.

---

## 📏 Working rules

Five people, one set of results. These are what kept them comparable:

1. **Never invent a validation split.** Everyone evaluates on the frozen files in `splits/`.
2. **macro-F1 is primary**, reported alongside accuracy and a confusion matrix.
3. **No external pretrained weights in any shipped model.** Pretrained networks appear only as
   labelled reference comparisons.
4. **Datasets and training weights stay out of Git** — only the shipped checkpoints are committed.
5. Prediction files keep the issued format exactly: `id,gender,articleType,season,usage`.

---

## 👥 Team

Built by **Team SG_G3** for COSC2753 Machine Learning at RMIT University, supervised by
Dr. Nguyen Thien Bao.

Le Duc Huy · Nguyen Gia Khang · Le Hoang Dang Khoa · Tran Hoang Nguyen · Nguyen Doan Trung Truc

---

## 📚 Acknowledgements

Dataset provided through RMIT Canvas for **COSC2753 Machine Learning**, educational use only, and
not redistributed here. Methods build on published work cited in full in each notebook's reference
section — among them k-reciprocal re-ranking (Zhong et al., 2017), ArcFace (Deng et al., 2019),
supervised contrastive learning (Khosla et al., 2020), multi-similarity loss (Wang et al., 2019),
logit adjustment (Menon et al., 2021) and FashionCLIP (Chia et al., 2022).

> This repository is university coursework, published as a record of the work — not as a solution to
> copy. If you are taking COSC2753, submitting any part of it as your own is academic misconduct.
