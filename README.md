# UCE_senescent_cells

This repository contains the code files used for the master´s thesis project: *Characterization of senescent cells using the Universal Cell Embeddings model (UCE) with single-cell RNA sequencing data*. The datasets, UCE output and other files are stored in a google Drive folder due to their size, and can be accessed here: [Google Drive Folder](https://drive.google.com/drive/folders/1BR6MiHFtKXx8H6wRLqGsK2TGhRMWu3Do?usp=drive_link).

All the files are organized as follow:

- **Datasets**: the original datasets containing the raw data (sk1, sk3, lu1 and bm1). Stored in [Google Drive Folder](https://drive.google.com/drive/folders/1BR6MiHFtKXx8H6wRLqGsK2TGhRMWu3Do?usp=drive_link).
- **Preprocessing**:
  - preprocess_S1: code used to pre-process the data from the **S1** testing sample.
  - preprocess_ALL: code used to pre-process the data from the **ALL** concatenated selected datasets.
- **Anndatas**: stored in the [Google Drive Folder](https://drive.google.com/drive/folders/1BR6MiHFtKXx8H6wRLqGsK2TGhRMWu3Do?usp=drive_link).
  - S1_adata.h5ad: anndata object, output from preprocess_S1.
  - ALL_adata.h5ad: anndata object, output from preprocess_ALL.
- **UCE_output**: stored in the [Google Drive Folder](https://drive.google.com/drive/folders/1BR6MiHFtKXx8H6wRLqGsK2TGhRMWu3Do?usp=drive_link). This folder contains the output files from UCE, both for the S1 and the ALL datasets.
- **Analysis**: clustering.py file with the code for the clustering analysis and UMAP visualization.
