'''This code subsets the senescent cells and reclusters them to gain insights about their subpopulations'''

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import gseapy as gp
from funciones import *
import colorcet as cc

# Load adata
adata = sc.read_h5ad('./aLL_cache/08_adata_n30_r1.0.h5ad')

# Reduce redundancy of the celltypist labels
adata.obs['celltypist_labels'] = adata.obs['celltypist_labels'].map(map_cell_types)

# Subset the adata with only senescent cells
sen_adata = adata[adata.obs['Sen_core'] == 'senescent']

# Compute neighbors graph and leiden clustering
print('\nClustering analysis for the senescent subset...')
sc.pp.neighbors(sen_adata, n_neighbors=30, use_rep='X_uce')
#sc.tl.leiden(sen_adata, resolution=0.2)

# Update cache
sen_adata.write('./ALL_cache/08_senescent.h5ad')
print("\nAdata saved successfully :)")

# Visualize clusters with UMAP
print('\nPreparing UMAP plots...')
colors = cc.glasbey[:30]

#sc.tl.umap(sen_adata)
#sc.pl.umap(sen_adata, color='leiden', title='Sen_core - clusters', save='_08_Sen_core_clusters.png') # Leiden clusters
#sc.pl.umap(sen_adata, color='celltypist_labels', title='Sen_core - cell types', palette=colors, save='_08_Sen_core_celltypes.png') # cell types
#sc.pl.umap(sen_adata, color='tissue', title='Sen_core - tissues', save='_08_Sen_core_tissues.png') # tissues




#### COMPARISON BETWEEN SKIN_1 AND SKIN_2 ####
print('\nSKIN_1 VS SKIN_2 ANALYSIS...')

# Subset the data
skin_cells = sen_adata[sen_adata.obs['tissue'].isin(['skin_1', 'skin_2'])]

# DEA
degs_skin={}  # dictionary to store the DEGs for each tissue

print('\nComputing DEA...')
sc.tl.rank_genes_groups(skin_cells, groupby='tissue', method='t-test')
#sc.pl.rank_genes_groups(skin_cells, n_genes=20, sharey=False, save='DEA_skin1_skin2.png') # plot the top 20 genes


for tissue in ['skin_1', 'skin_2']:
  degs = sc.get.rank_genes_groups_df(skin_cells, group=tissue) # Get the DEGs in a dataframe
  #degs.to_excel(f'DEA_{tissue}.xlsx', index=False) # Save DEGs to excel
  degs_skin[tissue] = degs

#DEA_dotplot(skin_cells, degs_skin, groupby='tissue', n=6, png_name='_DEGs_skin1_skin2.png')

# ORA
for tissue, degs_df in degs_skin.items():
  out_excel = f'ORA/ORA_{tissue}.xlsx'
  out_IDs = f'ORA/GO_IDs_{tissue}.xlsx'
  compute_ora(degs_df, out_excel, out_IDs)

# GSEA
for tissue, degs_df in degs_skin.items():
  outdir = f'GSEA_{tissue}'
  compute_gsea(degs_df, outdir, top_n=2000)



#### COMPARISON BETWEEN ALL TISSUES ####
# DEA
degs_dict={}  # dictionary to store the DEGs for each tissue

# Replace the skin_1 and skin_2 labels for 'skin'
sen_adata.obs['tissue'] = sen_adata.obs['tissue'].replace({'skin_1': 'skin', 'skin_2': 'skin'})

# Differential expression on each tissue
print('\nComputing DEA for each tissue...')
sc.tl.rank_genes_groups(sen_adata, groupby='tissue', method='t-test')
#sc.pl.rank_genes_groups(sen_adata, n_genes=20, sharey=False, save='DEA_tissues.png') # plot the top 20 genes

for tissue in ['skin', 'lungs', 'bone_marrow']:
  degs = sc.get.rank_genes_groups_df(sen_adata, group=tissue) # Get the DEGs for each tissue
  #degs.to_excel(f'DEA_{tissue}.xlsx', index=False) # Save DEGs to excel
  degs_dict[tissue] = degs # add the df to the dictionary

#DEA_dotplot(sen_adata, degs_dict, groupby='tissue', n=6, png_name='_DEGs_tissues.png')

# ORA
for tissue, degs_df in degs_dict.items():
  out_excel = f'ORA/ORA_{tissue}.xlsx'
  out_IDs = f'ORA/GO_IDs_{tissue}.xlsx'
  compute_ora(degs_df, out_excel, out_IDs)

# GSEA
for tissue, degs_df in degs_dict.items():
  outdir = f'GSEA_{tissue}'
  compute_gsea(degs_df, outdir, top_n=2000)