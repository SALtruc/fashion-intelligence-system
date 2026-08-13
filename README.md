# Machine Learning Assignment 2 — Fashion Intelligence System

COSC2753 Machine Learning · Assignment 2 (2026B) · RMIT
**Due: Sat 12 Sep 2026, 23:59 (Canvas)** — worth 40% of the course.

## Repo structure

```
Dataset/          # FashionDataset (train 38,617 rows / test 5,829 images) — committed for easy Colab clone
notebooks/        # eda.ipynb, task1_*.ipynb ... task4_*.ipynb
src/              # shared code: data loading, split, metrics
splits/           # fixed stratified train/val split — EVERYONE evaluates on this
report/           # report drafts (max 5 pages + 2 appendix, font 11)
predictions/      # prediction CSVs in styles_prediction.csv format
```

## Ground rules

1. **Evaluate on the shared split in `splits/`** — never make your own val split, or results aren't comparable.
2. **Metrics: macro-F1 + accuracy + confusion matrix** (data is heavily imbalanced — accuracy alone lies).
3. **No pretrained weights** in submitted models (ImageNet etc. only allowed for comparison).
4. **Model weights are NOT committed** (gitignored) — share via Drive.
5. Log every experiment in the shared experiments sheet so it can go in the report comparison table.
6. Prediction files must keep the exact `styles_prediction.csv` format: `id,gender,articleType,season,usage`.

## Setup (Colab)

```
!git clone https://github.com/SALtruc/Machine-Learning-Assignment-2.git
%cd Machine-Learning-Assignment-2
!pip install -q -r requirements.txt
```

> Dataset is for educational use in this course only. Keep this repo **private**.
