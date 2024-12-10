'''This code creates a consensus senescence signature and labels each cell as senescent, unknow or non-senescent'''

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from age_mapping import *


#### LOAD DATA ####
adata = sc.read_h5ad('./ALL_cache/03_adata_n30_r1.0.h5ad')


#### DICOTOMIC LABELING OF SENESCENT CELLS ####
# Label the cells for each signature
print('labeling the cells for each signature...')
signature_names = ['SenMayo', 'CoreScence', 'Casella', 'SenTissue']

for sig in signature_names: 
  print(f'\nSignature {sig}')

  # Calculate quantiles
  q90 = adata.obs[sig].quantile(0.90)
  q95 = adata.obs[sig].quantile(0.95)

  # Possible labels
  choices = ['senescent', 'unknown', 'non_senescent']

  # Set the conditions to label the cell as senescent, unknown or non_senescent
  conditions = [
    adata.obs[sig] > q95, # If score in quantile >95 -> senescent
    (adata.obs[sig] > q90) & (adata.obs[sig] <= q95), # If score in quantile 90 < x < 95 -> unknown
    adata.obs[sig] <= q90 # If score in quantile <90 -> non_senescent
  ]
  # Use numpy .select() to select the label based on the conditions
  adata.obs[f'{sig}_senescent'] = np.select(conditions, choices, default='non_senescent')

# convert the column to category type
adata.obs[f'{sig}_senescent'] = adata.obs[f'{sig}_senescent'].astype('category') 



#### CREATE A CONSENSUS LABEL ####
# Initialize an empty list to store the consensus labels
consensus_labels = []

# Iterate over all cells
print('\nCreating consensus labels...')
for idx, cell in adata.obs.iterrows():
  # For the current cell, collect the labels from the 4 signatures in a list
  labels = [cell[f'{sig}_senescent'] for sig in signature_names]

  # Count 'senescent' and 'unknown' labels
  senescent_count = labels.count('senescent')
  unknown_count = labels.count('unknown')

  # Apply consensus rules
  if senescent_count >= 2: # If cell is labeled as senescent in 2 or more signatures -> senescent
    consensus_labels.append('senescent')
  elif senescent_count == 1: # If cell is labeled as senescent in 1 signature -> unknown
    consensus_labels.append('unknown')
  elif unknown_count >= 1: # If cell is not labeled senescent by any signature, but unknown by at least 1 signature -> unknown
    consensus_labels.append('unknown')
  else: # else -> non senescent
    consensus_labels.append('non_senescent')

# Add consensus labels to adata.obs
adata.obs['Sen_consensus'] = consensus_labels
print('...done!\n')

# Save the cache
adata.write('./ALL_cache/04_adata_n30_r1.0.h5ad')
print('\nAdata successfully saved :)')



#### VISUALIZATION ####
# Calculate counts and percentages of detected senescent cells
label_counts = adata.obs['Sen_consensus'].value_counts()
label_percentages = (label_counts / len(adata.obs)) * 100

print('RESULTS:')
print('Counts:\n', label_counts)
print('\nPercentages:\n', label_percentages)

# UMAP plot colored by Sen_consensus
print('\nPlotting...')
sc.pl.umap(adata, color='Sen_consensus', cmap='viridis', title='Senescent cells consensus', save='_consensus.png')

# Plot percentage of SnCs by age in bone marrow (functions from age_mapping.py)
adata.obs['age'] = adata.obs['batch'].map(age_mapping)
get_age_plots(adata, 'bone_marrow')