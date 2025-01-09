'''This code visualizes and validates the identified senescent cells by the consensus signature'''

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import gseapy as gp
from funciones import *



#### LOAD THE DATA ####
adata = sc.read_h5ad('./ALL_cache/04_adata_n30_r1.0.h5ad')



#### VISUALIZATION ####
# Calculate counts and percentages of detected senescent cells
label_counts = adata.obs['Sen_consensus'].value_counts()
label_percentages = (label_counts / len(adata.obs)) * 100

print('RESULTS:\n')
print('Counts:\n', label_counts)
print('\nPercentages:\n', label_percentages)

# UMAP plot colored by Sen_consensus
print('\nPlotting...')
sc.pl.umap(adata, color='Sen_consensus', cmap='viridis', title='Senescent cells consensus', save='_consensus.png')

# Plot percentage of SnCs by age in bone marrow (function from funciones.py)
adata.obs['age'] = adata.obs['batch'].map(age_map)
get_age_plots(adata, 'bone_marrow', 'Sen_consensus')




#### DEA ####
# Differential expression: Senescent vs Non-Senescent
sc.tl.rank_genes_groups(adata, groupby='Sen_consensus', reference='non_senescent', method='t-test')
sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False, save='DEA_Sen_consensus.png') # plot the top 20 genes

# get DEGs
degs = sc.get.rank_genes_groups_df(adata, group='senescent') # Get the DEGs from senescent group
degs.to_excel(f'DEA_sen_consensus.xlsx', index=False) # Save DEGs to excel


# Update cache
adata.write('./ALL_cache/05_adata_n30_r1.0.h5ad')
print("\nAdata saved in /ALL_cache")

# Marker gene expression
print("\nMarker gene expression:\n")
senescence_markers = ['CDKN1A', 'CDKN2A', 'IL6', 'MMP1', 'MMP3', 'TNF']
marker_expr = adata[:, senescence_markers].to_df().groupby(adata.obs['Sen_consensus'], observed=False).mean()
print(marker_expr)


#### ORA ####
compute_ora(degs, out_excel='ORA/ORA_Sen_consensus.xlsx', out_IDs='ORA/GO_IDs_sen_consensus.xlsx')


#### GSEA ####
compute_gsea(degs, 'GSEA_Sen_consensus', top_n=2000)