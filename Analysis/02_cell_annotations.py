'''This code uses celltypist models to identify the cell types and label them'''

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import celltypist
from celltypist import models
import colorcet as cc
from funciones import *


#### LOAD AND PREPARE THE DATA ####
# Load the adata generated in the clustering analysis
adata = sc.read_h5ad('./ALL_cache/adata_n30_r1.0.h5ad')

# Normalization and log-transformation
sc.pp.normalize_total(adata, target_sum=1e4) # Normalising
adata.X = np.log1p(adata.X)  # Log transformation
print('\nData normalized and log-transformed.')



#### CELL ANNOTATIONS ####
# Subset the data from each tissue
print('\nSubsetting the adata based on tissues...')
skin_cells = adata[adata.obs['tissue'].isin(['skin_1', 'skin_2'])]
bm_cells = adata[adata.obs['tissue'] == 'bone_marrow']
lung_cells = adata[adata.obs['tissue'] == 'lungs']

# Run CellTypist models on each subset
print('\nDownloading CellTypist models:\n')
models.download_models(force_update = True)

print('\n#1 Computing skin predictions...')
predictions_skin = celltypist.annotate(skin_cells, model='Adult_Human_Skin.pkl', majority_voting=True, mode='best match')
print('\n#2 Computing bone marrow predictions...')
predictions_bm = celltypist.annotate(bm_cells, model='Immune_All_High.pkl', majority_voting=True, mode='best match')
print('\n#3 Computing lung predictions...')
predictions_lung = celltypist.annotate(lung_cells, model='Human_Lung_Atlas.pkl', majority_voting=True, mode='best match')

# Add predictions to each adata.obs
print('\nAdding predictions to each subset-adata')
predictions_skin.to_adata(skin_cells)
predictions_bm.to_adata(bm_cells)
predictions_lung.to_adata(lung_cells)

# Merge predictions on the original adata
## Create a new column for cell type annotations and initialize with NaN values
print('\nMerging the predictions to the original adata...')
adata.obs['celltypist_labels'] = np.nan  

## Load the celltypist_labels on the new column
adata.obs.loc[skin_cells.obs.index, 'celltypist_labels'] = skin_cells.obs['predicted_labels']
adata.obs.loc[bm_cells.obs.index, 'celltypist_labels'] = bm_cells.obs['predicted_labels']
adata.obs.loc[lung_cells.obs.index, 'celltypist_labels'] = lung_cells.obs['predicted_labels']





#### SAVING RESULTS ####
adata.write('./ALL_cache/02_adata_n30_r1.0.h5ad')
print('\nAdata successfully saved :)')




#### VISUALIZATION ####
colors = cc.glasbey[:30]

# UMAP plots per tissue
print('\nSaving UMAP plots...')
sc.pl.umap(skin_cells, color='leiden_n30_r1.0', title='Skin Leiden clusters', palette=colors, save='_SKIN_clusters.png')
sc.pl.umap(skin_cells, color='predicted_labels', title='Skin Cell Types', palette=colors, save='_SKIN_cell_types.png')

sc.pl.umap(bm_cells, color='leiden_n30_r1.0', title='Bone Marrow Leiden clusters', palette=colors, save='_MARROW_clusters.png')
sc.pl.umap(bm_cells, color='predicted_labels', title='Bone Marrow Cell Types', palette=colors, save='_MARROW_cell_types.png')

sc.pl.umap(lung_cells, color='leiden_n30_r1.0', title='Lung Leiden clusters', palette=colors, save='_LUNG_clusters.png')
sc.pl.umap(lung_cells, color='predicted_labels', title='Lung Cell Types', palette=colors, save='_LUNG_cell_types.png')

