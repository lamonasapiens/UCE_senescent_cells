'''This code visualizes and validates the identified senescent cells by the SenCore signature
(same as 05_Sen_consensus_validation.py)'''

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import gseapy as gp
from funciones import *



#### LOAD THE DATA ####
adata = sc.read_h5ad('./ALL_cache/07_adata_n30_r1.0.h5ad')



#### VISUALIZATION ####
# Calculate counts and percentages of detected senescent cells
label_counts = adata.obs['Sen_core'].value_counts()
label_percentages = (label_counts / len(adata.obs)) * 100

print('RESULTS:\n')
print('Counts:\n', label_counts)
print('\nPercentages:\n', label_percentages)

# UMAP plot colored by Sen_consensus
print('\nPlotting...')
sc.pl.umap(adata, color='Sen_core', cmap='viridis', title='Sen_core gradient plot', save='_Sen_core.png')

# Plot percentage of SnCs by age in bone marrow (function from funciones.py)
adata.obs['age'] = adata.obs['batch'].map(age_map)
get_age_plots(adata, 'bone_marrow', 'Sen_core')




#### DEA ####
# Differential expression: Senescent vs Non-Senescent
sc.tl.rank_genes_groups(adata, groupby='Sen_core', reference='non_senescent', method='t-test')
sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False, save='DEA_sen_core.png') # plot the top 20 genes

# get DEGs
degs = sc.get.rank_genes_groups_df(adata, group='senescent') # Get the DEGs in a dataframe
degs.to_excel(f'DEA_sen_core.xlsx', index=False) # Save DEGs to excel

# Update cache
adata.write('./ALL_cache/08_adata_n30_r1.0.h5ad')
print("\nAdata saved in /ALL_cache")

# Filter out the unknown cells
sen_adata = adata[adata.obs['Sen_core'].isin(['senescent', 'non_senescent'])]

# Marker gene expression
sc.tl.rank_genes_groups(sen_adata, groupby='Sen_core', method='t-test')

gene_set = ['CDKN1A', 'CDKN2A', 'CCL5', 'CXCL8', 'TIMP1', 'LMNB1', 'MKI67', 'PARP1']

fig, ax = plt.subplots(1,1, figsize=(5, 5))
sc.pl.dotplot(sen_adata, groupby='Sen_core', var_names=gene_set, show=True, swap_axes=True, ax=ax, save='dotplot_sen_core.png')


#### ORA ####
compute_ora(degs, out_excel='ORA/ORA_Sen_core.xlsx', out_IDs='ORA/GO_IDs_sen_core.xlsx')


#### GSEA ####
compute_gsea(degs, 'GSEA_Sen_core', top_n=2000)