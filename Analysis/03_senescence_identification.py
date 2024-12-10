'''This code uses several senescence signatures to identify potential senescent cells on the dataset'''

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

#### LOAD DATA ####
adata = sc.read_h5ad('./ALL_cache/02_adata_n30_r1.0.h5ad')


#### PREPARING SENESCENCE SIGNATURES ####
# Load the senescence signatures from their respective .txt files
paths = ['Casella_UP', 'Casella_DOWN', 'CoreScence_UP', 'CoreScence_DOWN', 'SenMayo', 'SenSkin_UP', 'SenSkin_DOWN', 'SenLungs_UP', 'SenLungs_DOWN', 'SenMarrow_UP', 'SenMarrow_DOWN']

signatures = {} # create a dictionary to store the signatures

for i in paths:
  file_path = f'./Sen_Signatures/{i}.txt'
  with open(file_path, "r") as file:
    signatures[i] = file.read().split()
print('\nSignatures loaded')


# Function to calculate enrichment and store gene scores
def calculate_signature_score(adata, up_genes, down_genes=None, signature="signature_name"):
  '''This function applies a signature to calculate the score of upregulated genes and substracts the score of downreguated genes'''
  
  sc.tl.score_genes(adata, gene_list=up_genes, score_name=signature)
  
  if down_genes: # If downregulated genes are provided, substract this score
    sc.tl.score_genes(adata, gene_list=down_genes, score_name=f'{signature}_down')
    adata.obs[signature] = adata.obs[signature] - adata.obs[f'{signature}_down']
  return adata



#### APPLY GLOBAL SIGNATURES ####
# Apply SenMayo, Casella, and CoreScence signatures
print('\nCalculating scores for SenMayo...')
calculate_signature_score(adata, signatures['SenMayo'], down_genes=None, signature="SenMayo")
print('\nCalculating scores for Casella...')
calculate_signature_score(adata, signatures['Casella_UP'], signatures['Casella_DOWN'], signature="Casella")
print('\nCalculating scores for CoreScence...')
calculate_signature_score(adata, signatures['CoreScence_UP'], signatures['CoreScence_DOWN'], signature="CoreScence")



#### APPLY TISSUE-SPECIFIC SIGNATURES ####
# Subset the data from each tissue
print('\nSubsetting the adata based on tissues...')
skin = adata[adata.obs['tissue'].isin(['skin_1', 'skin_2'])].copy()
bm = adata[adata.obs['tissue'] == 'bone_marrow'].copy()
lung = adata[adata.obs['tissue'] == 'lungs'].copy()

# Calculate scores for each subset
print('\nCalculating scores for SenTissue...')
calculate_signature_score(skin, signatures['SenSkin_UP'], signatures['SenSkin_DOWN'], signature="SenTissue")
calculate_signature_score(bm, signatures['SenMarrow_UP'], signatures['SenMarrow_DOWN'], signature="SenTissue")
calculate_signature_score(lung, signatures['SenLungs_UP'], signatures['SenLungs_DOWN'], signature="SenTissue")

# Normalize scores for each subset so that they can be comparable
print('\nNormalizing...\n')
for adata_tissue in [skin, bm, lung]:
  scaler = StandardScaler()
  adata_tissue.obs['normalized'] = scaler.fit_transform(adata_tissue.obs[['SenTissue']])
    
# Merge the adatas (keeping only the normalized scores)
print('\nMerging the adatas...')
merged = pd.concat([
  skin.obs[['normalized']],
  bm.obs[['normalized']],
  lung.obs[['normalized']]
])

# Upload the normalized scores into the original adata
adata.obs['SenTissue'] = merged['normalized'] 

# Check if there are missing values in the merged adata.obs['SenTissue']
print('\nMissing values in merged adata:')
print(adata.obs['SenTissue'].isnull().sum())

# Save the scores in cache
adata.write('./ALL_cache/03_adata_n30_r1.0.h5ad')
print('\nAdata saved in ALL_cache')



#### VISUALIZATION ####
# Generate gradient plots for each signature
print('\nPreparing gradient plots...')
signature_names = ['SenMayo', 'CoreScence', 'Casella', 'SenTissue']

for sig in signature_names:
  sc.pl.umap(adata, color=sig, cmap='viridis', title=f'{sig} Gradient Plot', save=f'_{sig}.png')


# Plot senescent score distributions
print('\nPreparing distribution plots...')
fig, axes = plt.subplots(2, 2, figsize=(10, 8)) # Fix 2 rows and 2 columns
axes = axes.flatten()

for i, sig in enumerate(signature_names):
  sns.histplot(adata.obs[sig], kde=True, ax=axes[i]) # Generate an histogram
  axes[i].axvline(adata.obs[sig].quantile(0.95), color='red', linestyle='--', label='95th percentile') # Red line in the 95th percentile
  axes[i].legend()
  axes[i].set_title(f'{sig} Score Distribution')

plt.tight_layout() # Adjust layout
plt.savefig('Figures/signatures_score_distributions.png') # Save the image