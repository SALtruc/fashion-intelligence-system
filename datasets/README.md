# Dataset folder

This folder contains the raw FashionDataset files used by the assignment:

```text
datasets/
├── train/
│   ├── images_train/
│   └── styles_train.csv
└── test/
    ├── images_test/
    └── styles_test.csv
```

The CSV files and image folders are not included in the repository because they are large and intended to be shared through Drive. Keeping these files out of Git also prevents large binary files from making the repository unnecessarily large.

## Download and setup

1. Download `A2_FashionDataset.zip` from [Google Drive](https://drive.google.com/file/d/1OpTIeDsvhxDlpI3UdCb60WKHma8MHjnX/view?usp=sharing).
2. Extract the zip file. It should contain a `FashionDataset/` folder.
3. Copy the `train/` and `test/` folders from the extracted `FashionDataset/` folder into this `datasets/` folder.

After setup, the repository should contain `datasets/train/` and `datasets/test/` with the original CSV files and image folders.

## Important: keep raw data unchanged

All downloaded files in this folder are raw dataset files. Do not modify them under any circumstances. This includes renaming, deleting, overwriting, preprocessing, resizing, augmenting, relabelling, or splitting the files.

Perform preprocessing, duplicate checks, data splitting, and other experiments in separate code or output locations. The raw files must remain available as the unmodified source data for every experiment.
