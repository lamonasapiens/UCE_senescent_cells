'''This code filters the highly variable genes and senescent genes, and corrects for cell type. Then it performs DEA and GSEA 
to get a list of differentialy expressed genes that are not cofounded by cell-type'''

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import gseapy as gp
from funciones import *
import colorcet as cc
import seaborn as sns
from sklearn.decomposition import PCA



# Load adata
adata = sc.read_h5ad('./aLL_cache/05_adata_n30_r1.0.h5ad')

# Filter out the unknown cells
adata = adata[adata.obs['Sen_consensus'].isin(['senescent', 'non_senescent'])]



#### SELECT HIGHLY VARIABLE GENES ####
# Identify highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=3000)

# Include the genes from the signatures
paths = ['Casella_UP', 'Casella_DOWN', 'CoreScence_UP', 'CoreScence_DOWN', 'SenMayo', 'SenSkin_UP', 'SenSkin_DOWN', 'SenLungs_UP', 'SenLungs_DOWN', 'SenMarrow_UP', 'SenMarrow_DOWN']
signatures = set() # create a set to store the signatures

# Loop through the paths to load the signatures
for i in paths:
  file_path = f'./Sen_Signatures/{i}.txt'
  with open(file_path, "r") as file:
    signature = file.read().split()
    signatures.update(signature) # Add genes to the set (avoids duplicates)

# Include the senescent genes into the highly variable genes
senescence_genes = [gene for gene in signatures if gene in adata.var_names]
adata.var['highly_variable'] = adata.var['highly_variable'] | adata.var_names.isin(senescence_genes)

# Subset
adata = adata[:, adata.var['highly_variable']]

print('\nAdata filtered with highly variable genes and senescent genes')



#### BATCH CORRECTIONS ####
# Count cells in each category
category_counts = adata.obs['celltypist_labels'].value_counts()

# Remove low-count categories (threshold = 10)
threshold = 10
valid_categories = category_counts[category_counts >= threshold].index
adata = adata[adata.obs['celltypist_labels'].isin(valid_categories)]

# Apply cell-type batch correction
print('\nApplying batch (cell-type) corrections...')
sc.pp.combat(adata, key='celltypist_labels')



#### DEA ####
print('\nComputing DEA...')
# Differential expression: Senescent vs Non-Senescent
sc.tl.rank_genes_groups(adata, groupby='Sen_consensus', reference='non_senescent', method='t-test')
sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False, save='DEA_Sen_corrected.png') # plot the top 20 genes

# get DEGs from senescent cells
degs = sc.get.rank_genes_groups_df(adata, group='senescent') 
degs.to_excel(f'DEA_sen_corrected.xlsx', index=False)


# Update cache
adata.write('./ALL_cache/06_adata_n30_r1.0.h5ad')
print("\nAdata saved in /ALL_cache")



#### GSEA ####
compute_gsea(degs, out_excel='GSEA_Sen_corrected.xlsx', out_IDs='GO_IDs_corrected.xlsx')

