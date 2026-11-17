# Data folder (not tracked by git)

Dataset: **LGG MRI Segmentation** (Kaggle, `mateuszbuda/lgg-mri-segmentation`, derived from TCGA-LGG).
Download and unzip it so the `.tif` files are somewhere below this folder, e.g.:

```
data/
└── kaggle_3m/
    ├── TCGA_CS_4941_19960909/
    │   ├── TCGA_CS_4941_19960909_1.tif        # 3-channel MRI slice (pre-contrast, FLAIR, post-contrast)
    │   ├── TCGA_CS_4941_19960909_1_mask.tif   # binary tumour mask
    │   └── ...
    └── ...
```
Files are found recursively, so the exact nesting does not matter.
