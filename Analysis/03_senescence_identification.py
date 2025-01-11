'''This code uses several senescence signatures to identify potential senescent cells on the dataset. The signatures are
applied to each tissue separatedly to avoid bias.'''

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from funciones import calculate_signature_score

#### LOAD DATA ####
adata = sc.read_h5ad('./ALL_cache/02_adata_n30_r1.0.h5ad')



#### PREPARING SENESCENCE SIGNATURES ####
# Load the senescence signatures from their respective .txt files
paths = ['Casella_UP', 'Casella_DOWN', 'CoreScence_UP', 'CoreScence_DOWN', 'SenMayo', 'SenSkin_UP', 'SenSkin_DOWN', 'SenLungs_UP', 'SenLungs_DOWN', 'SenMarrow_UP', 'SenMarrow_DOWN']

signatures = {} # create a dictionary to store the signatures

# Loop through the paths to load the signatures
for i in paths:
  file_path = f'./Sen_Signatures/{i}.txt'
  with open(file_path, "r") as file:
    signatures[i] = file.read().split()
print('\nSignatures loaded')



#### APPLY THE SIGNATURES ####
# Subset the adata for each tissue
skin = adata[adata.obs['tissue'].isin(['skin_1', 'skin_2'])].copy()
bm = adata[adata.obs['tissue'] == 'bone_marrow'].copy()
lung = adata[adata.obs['tissue'] == 'lungs'].copy()

# Iterate to apply the senescent signatures to each tissue
signatures_names = ["SenMayo", "Casella", "CoreScence", "SenTissue"]
tissues = ["Skin", "Marrow", "Lungs"]
adata_list = [skin, bm, lung]

for i in range(len(adata_list)):
  print(f'\nCalculating scores for {tissues[i]}:')
  print('SenMayo...')
  calculate_signature_score(adata_list[i], signatures['SenMayo'], down_genes=None, signature="SenMayo")
  print('Casella...')
  calculate_signature_score(adata_list[i], signatures['Casella_UP'], signatures['Casella_DOWN'], signature="Casella")
  print('CoreScence...')
  calculate_signature_score(adata_list[i], signatures['CoreScence_UP'], signatures['CoreScence_DOWN'], signature="CoreScence")
  print('SenTissue...')
  calculate_signature_score(adata_list[i], signatures[f'Sen{tissues[i]}_UP'], signatures[f'Sen{tissues[i]}_DOWN'], signature="SenTissue")




#### SAVE THE ADATAS IN CACHE ####
for i in range(len(adata_list)):
  adata_list[i].write(f'./ALL_cache/03_{tissues[i]}.h5ad')
  
print('\nAdatas saved in ALL_cache')



#### VISUALIZATION ####
# Generate gradient plots for each signature
print('\nPreparing gradient plots...')
for i in range(len(adata_list)):
  for sig in signatures_names:
    sc.pl.umap(adata_list[i], color=sig, cmap='viridis', title=f'{tissues[i]} {sig} Gradient Plot', save=f'_{tissues[i]}_{sig}.png')


# Plot senescent score distributions for each tissue
print('\nPreparing distribution plots...')

for i in range(len(adata_list)): # Iterate through the tissues
  fig, axes = plt.subplots(2, 2, figsize=(10, 8)) # Fix 2 rows and 2 columns
  axes = axes.flatten()

  for j, sig in enumerate(signatures_names): # Iterate through the signatures
    sns.histplot(adata_list[i].obs[sig], kde=True, ax=axes[j]) # Generate a histogram
    axes[j].axvline(adata_list[i].obs[sig].quantile(0.95), color='red', linestyle='--', label='95th percentile') # Red line in the 95th percentile
    axes[j].legend()
    axes[j].set_title(f'{sig} Score Distribution')

  plt.tight_layout() # Adjust layout
  plt.savefig(f'Figures/{tissues[i]}_score_distributions.png') # Save the image
  plt.close(fig) 
