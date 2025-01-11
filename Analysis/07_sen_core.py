'''This code applies a new senescent signature called SenCore to the original dataset. SenCore has the top 20 DEGs found
in the previous step (06_corrected_signature.py)'''

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import gseapy as gp
import joblib
import colorcet as cc
from funciones import *

#### LOAD DATA ####
adata = sc.read_h5ad('./ALL_cache/05_adata_n30_r1.0.h5ad')


#### APPLY THE CORRECTED SIGNATURE: SenCore ####
# Load the senescent genes
with open('./Sen_Signatures/core_genes.txt', "r") as file:
  genes = file.read().split()

# Subset the adata for each tissue to avoid bias
skin = adata[adata.obs['tissue'].isin(['skin_1', 'skin_2'])].copy()
bm = adata[adata.obs['tissue'] == 'bone_marrow'].copy()
lung = adata[adata.obs['tissue'] == 'lungs'].copy()

# Apply the Sen_core signature
tissues = ["Skin", "Marrow", "Lungs"]
adata_list = [skin, bm, lung]

for i in range(len(adata_list)):
  print(f'\nCalculating scores for {tissues[i]}:')
  calculate_signature_score(adata_list[i], genes, down_genes=None, signature="Sen_core")




#### CONVERT THE SCORES INTO CATEGORICAL LABELS ####
# Label cells according to cell cycle
label_arrested_cells(skin)
label_arrested_cells(bm)
label_arrested_cells(lung)

# Label the cells as senescent, unknown and non_senescent for each adata
for adata_tissue in adata_list:
  # Calculate quantiles
  q90 = adata_tissue.obs['Sen_core'].quantile(0.90)
  q70 = adata_tissue.obs['Sen_core'].quantile(0.70)

  # Possible labels
  choices = ['senescent', 'unknown', 'non_senescent']

  # Set the conditions to label the cell as senescent or non_senescent
  conditions = [
    adata_tissue.obs['Sen_core'] > q90, # If score in quantile >90 -> senescent
    (adata_tissue.obs['Sen_core'] > q70) & (adata_tissue.obs['Sen_core'] <= q90), # If score in quantile 70 < x < 90 -> unknown
    adata_tissue.obs['Sen_core'] <= q70 # If score in quantile <70 -> non_senescent
  ]
  # Use numpy .select() to select the label based on the conditions
  adata_tissue.obs['Sen_core'] = np.select(conditions, choices, default='non_senescent')

  # If cells are not in cell cycle arrest, label them as non_senescent
  adata_tissue.obs.loc[adata_tissue.obs['cell_cycle_arrest'] == False, 'Sen_core'] = 'non_senescent'

  # convert the column to category type
  adata_tissue.obs['Sen_core'] = adata_tissue.obs['Sen_core'].astype('category') 




# LOAD THE SEN_CORE LABELS INTO THE ORIGINAL ADATA
# Extract 'Sen_core' labels from each tissue adata
skin_labels = skin.obs[['Sen_core']]
bm_labels = bm.obs[['Sen_core']]
lung_labels = lung.obs[['Sen_core']]

# Combine the labels into a single DataFrame
combined_labels = pd.concat([skin_labels, bm_labels, lung_labels])

# Ensure the indices in the combined_labels match the indices in the original adata
assert combined_labels.index.isin(adata.obs.index).all(), "Indices do not align!"

# Initialize the 'Sen_core' column in adata with 'non_senescent'
adata.obs['Sen_core'] = 'non_senescent' # (the non-arrested cells will remain with a non_senescent label)

# Add the 'Sen_core' labels back to the original adata
adata.obs.loc[combined_labels.index, 'Sen_core'] = combined_labels['Sen_core']



#### SAVE THE ADATA IN CACHE ####
adata.write('./ALL_cache/07_adata_n30_r1.0.h5ad')
print('\nAdata successfully saved :)')
