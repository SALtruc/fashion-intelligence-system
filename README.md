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

## Run the finalized EDA

Before running the notebook, confirm the training dataset has this structure:

```text
datasets/
└── train/
    ├── styles_train.csv
    └── images_train/
        ├── 1163.jpg
        └── ...
```

### Run interactively

Start Jupyter Notebook from the repository root:

```bash
uv run jupyter notebook
```

Open `notebooks/00_eda_and_preprocessing.ipynb`, select the project's `.venv` kernel if
prompted, then choose **Kernel → Restart Kernel and Run All Cells**. Press
`Ctrl+S` to save the cell outputs in the notebook.

The image validation and duplicate-detection sections process approximately
38,000 images, so a complete run may take several minutes.

### Run from the command line

To execute every cell in a fresh kernel and save the outputs to a separate
notebook, run the following from the repository root.

Windows PowerShell:

```powershell
uv run jupyter nbconvert `
  --to notebook `
  --execute "notebooks\00_eda_and_preprocessing.ipynb" `
  --output "00_eda_and_preprocessing_executed.ipynb" `
  --ExecutePreprocessor.timeout=-1
```

macOS or Linux:

```bash
uv run jupyter nbconvert \
  --to notebook \
  --execute notebooks/00_eda_and_preprocessing.ipynb \
  --output 00_eda_and_preprocessing_executed.ipynb \
  --ExecutePreprocessor.timeout=-1
```

The executed notebook is saved as
`notebooks/00_eda_and_preprocessing_executed.ipynb`. If execution immediately raises a
`FileNotFoundError`, verify that `datasets/train/styles_train.csv` and
`datasets/train/images_train/` exist.

> Dataset is for educational use in this course only. Keep this repo **private**.
