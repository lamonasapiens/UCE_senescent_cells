'''This code will generate a neighborhood graph and clustering analysis with Leiden using different parameters for the
variables n_neighbors_list and leiden_res (resolution). It creates a cache to store all the generated adatas and UMAP
plots for visualization'''

import os
import numpy as np
import scanpy as sc
import warnings
from funciones import assign_tissue, run_clustering

# Ignore warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*leiden.*")



##### CLUSTERING ANALYSIS #####
# Load data
ALL = sc.read_h5ad('../Datasets/UCE_output/ALL/ALL_adata_uce_adata.h5ad')
print('\nData ALL loaded')

# Apply the function "assign_tissue" from funciones.py to create the 'tissue' column
ALL.obs['tissue'] = ALL.obs['batch'].apply(assign_tissue)
print('\nCreated the "Tissue" column')

# Parameters to explore:
n_neighbors_list = [15, 30]
leiden_res = [0.5, 1.0, 1.5]

# Create a dictionary to store all the adatas
results = {}

# Perform clustering analysis looping on the parameters to explore, using our function from funciones.py:
for n in n_neighbors_list:
  for r in leiden_res:
    result_key = f'n{n}_r{r}'
    results[result_key] = run_clustering(ALL.copy(), n, r, cache_dir='ALL_cache')



#### VISUALIZE RESULTS ####
# Plot UMAPs
print('\nGenerating UMAP plots...')
for result_key, result_adata in results.items():
  sc.pl.umap(result_adata, color=f'leiden_{result_key}', title=f'Clustering {result_key}', save=f'_ALL_{result_key}.png', show=False)
  sc.pl.umap(result_adata, color='tissue', title=f'Tissues {result_key}', save=f'_ALL_{result_key}_tissues.png', show=False)
