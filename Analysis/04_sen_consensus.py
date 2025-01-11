'''This code creates a consensus senescence signature and labels each cell as senescent, unknow or non-senescent'''

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from funciones import *


#### LOAD DATA ####
skin = sc.read_h5ad('./ALL_cache/03_Skin.h5ad')
bm = sc.read_h5ad('./ALL_cache/03_Marrow.h5ad')
lung = sc.read_h5ad('./ALL_cache/03_Lungs.h5ad')


# Label the arrested cells based on their cell cycle markers
label_arrested_cells(skin)
label_arrested_cells(bm)
label_arrested_cells(lung)


#### DICOTOMIC LABELING OF SENESCENT CELLS ####
# Label the cells for each signature
print('labeling the cells for each signature...')
signature_names = ['SenMayo', 'CoreScence', 'Casella', 'SenTissue']

for adata in [skin, bm, lung]:
  for sig in signature_names: 

    # Calculate quantiles
    q75 = adata.obs[sig].quantile(0.75)
    q85 = adata.obs[sig].quantile(0.85)

    # Possible labels
    choices = ['senescent', 'unknown', 'non_senescent']

    # Set the conditions to label the cell as senescent, unknown or non_senescent
    conditions = [
      adata.obs[sig] > q85, # If score in quantile >85 -> senescent
      (adata.obs[sig] > q75) & (adata.obs[sig] <= q85), # If score in quantile 80 < x < 90 -> unknown
      adata.obs[sig] <= q75 # If score in quantile <80 -> non_senescent
    ]
    # Use numpy .select() to select the label based on the conditions
    adata.obs[f'{sig}_senescent'] = np.select(conditions, choices, default='non_senescent')

    # If cells are not in cell cycle arrest, label them as non_senescent
    adata.obs.loc[adata.obs['cell_cycle_arrest'] == False, f'{sig}_senescent'] = 'non_senescent'

    # convert the column to category type
    adata.obs[f'{sig}_senescent'] = adata.obs[f'{sig}_senescent'].astype('category') 




#### CREATE A CONSENSUS LABEL ####
print('\nCreating consensus labels...')

for adata in [skin, bm, lung]:
  # Initialize an empty list to store the consensus labels
  consensus_labels = []

  # Iterate over all cells
  for idx, cell in adata.obs.iterrows():
    # For the current cell, collect the labels from the 4 signatures in a list
    labels = [cell[f'{sig}_senescent'] for sig in signature_names]

    # Count 'senescent' and 'unknown' labels
    senescent_count = labels.count('senescent')
    unknown_count = labels.count('unknown')

    # Apply consensus rules
    if senescent_count >= 3: # If cell is labelled as senescent in 3 or more signatures -> senescent
      consensus_labels.append('senescent')
    elif senescent_count >= 1: # If cell is labelled as senescent in 1 or 2 signatures -> unknown
      consensus_labels.append('unknown')
    elif unknown_count >= 4: # If cell is not labelled as senescent by any signature, but unknown by 4 signatures -> unknown
      consensus_labels.append('unknown')
    else: # else -> non senescent
      consensus_labels.append('non_senescent')

  # Add consensus labels to adata.obs
  adata.obs['Sen_consensus'] = consensus_labels

print('...done!\n')



# LOAD THE CONSENSUS LABELS INTO THE ORIGINAL ADATA
adata = sc.read_h5ad('./ALL_cache/02_adata_n30_r1.0.h5ad')

# Extract 'Sen_consensus' labels from each tissue adata
skin_labels = skin.obs[['Sen_consensus']]
bm_labels = bm.obs[['Sen_consensus']]
lung_labels = lung.obs[['Sen_consensus']]

# Combine the labels into a single DataFrame
combined_labels = pd.concat([skin_labels, bm_labels, lung_labels])

# Ensure the indices in the combined_labels match the indices in the original adata
assert combined_labels.index.isin(adata.obs.index).all(), "Indices do not align!"

# Initialize the 'Sen_consensus' column in adata with 'non_senescent'
adata.obs['Sen_consensus'] = 'non_senescent' # (the non-arrested cells will remain with a non_senescent label)

# Add the 'Sen_consensus' labels back to the original adata
adata.obs.loc[combined_labels.index, 'Sen_consensus'] = combined_labels['Sen_consensus']

# Verify the labels have been added correctly
print(adata.obs['Sen_consensus'].head())



#### SAVE THE ADATA IN CACHE ####
adata.write('./ALL_cache/04_adata_n30_r1.0.h5ad')
print('\nAdata successfully saved :)')
