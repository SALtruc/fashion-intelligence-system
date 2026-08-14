# Machine Learning Assignment 2 — Fashion Intelligence System

COSC2753 Machine Learning · Assignment 2 (2026B) · RMIT
**Due: Sat 12 Sep 2026, 23:59 (Canvas)** — worth 40% of the course.

## Repo structure

```
datasets/         # FashionDataset (train 38,617 rows / test 5,829 images) — share via Drive
models/           # model weights (gitignored) — share via Drive
notebooks/        # eda.ipynb, task1_*.ipynb ... task4_*.ipynb
src/              # shared code: data loading, split, metrics
splits/           # fixed stratified train/val split — EVERYONE evaluates on this
predictions/      # prediction CSVs in styles_prediction.csv format
```

## Ground rules

1. **Datasets are NOT committed** (gitignored) — share via Drive.
2. **Evaluate on the shared split in `splits/`** — never make your own val split, or results aren't comparable.
3. **Metrics: macro-F1 + accuracy + confusion matrix** (data is heavily imbalanced — accuracy alone lies).
4. **No pretrained weights** in submitted models (ImageNet etc. only allowed for comparison).
5. **Model weights are NOT committed** (gitignored) — share via Drive.
6. Log every experiment in the shared experiments sheet so it can go in the report comparison table.
7. Prediction files must keep the exact `styles_prediction.csv` format: `id,gender,articleType,season,usage`.

## Setup

Clone the repository and change into the project directory:

```bash
git clone https://github.com/SALtruc/Machine-Learning-Assignment-2.git

cd Machine-Learning-Assignment-2
```

Before setting up the project, make sure [`uv`](https://docs.astral.sh/uv/getting-started/installation/) is installed globally and available on your `PATH`:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell
irm https://astral.sh/uv/install.ps1 | iex
```

Verify the installation, then run the following from the repository root. This creates `.venv` and installs the locked dependencies:

```bash
uv --version
uv sync --frozen
```

> Dataset is for educational use in this course only. Keep this repo **private**.
