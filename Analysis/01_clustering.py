'''This code will generate a neighborhood graph and clustering analysis with Leiden using different parameters for the
variables n_neighbors_list and leiden_res (resolution). It creates a cache to store all the generated adatas and UMAP
plots for visualization'''

import os
import numpy as np
import scanpy as sc
import warnings

# Ignore warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*leiden.*")


#### FUNCTIONS ####
# Function to label the cells based on their tissue origin (batch)
def assign_tissue(batch):
    if batch.startswith("sk1_S"):
        return "skin_1"
    elif batch == "sk3":
        return "skin_2"
    elif batch.startswith("bm1_B"):
        return "bone_marrow"
    elif batch == "lu1":
        return "lungs"
    else:
        return "unknown"


# Function for computing and cache
def run_clustering(adata, n_neighbors, res, cache_dir='cache'):
  '''This function generates a neighbors graph and computes leiden cluster analysis
  and UMAP space on a given adata object. It also saves the adata in a cache file.'''

  print(f'\n***** n_neighbors={n_neighbors}, resolution={res} *****')

  # Create cache directory if it doesn't exist
  os.makedirs(cache_dir, exist_ok=True)
  
  # Generate a unique cache filename
  cache_file = f'{cache_dir}/adata_n{n_neighbors}_r{res}.h5ad' 
  
  # Check if cached result exists
  if os.path.exists(cache_file):
    print(f'Loading cached result')
    return sc.read_h5ad(cache_file)
  
  else: # If there is no cache, perform clustering and compute UMAP
    print('Computing neighbors graph')
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X_uce')
    print('Computing Leiden clustering')
    sc.tl.leiden(adata, resolution=res, key_added=f'leiden_n{n_neighbors}_r{res}')
    print('Computing UMAP')
    sc.tl.umap(adata)
  
  # Save result to cache
  adata.write(cache_file)
  
  return adata



##### CLUSTER ANALYSIS #####
# Load data
ALL = sc.read_h5ad('../Datasets/UCE_output/ALL/ALL_adata_uce_adata.h5ad')
print('\nData ALL loaded')

# Apply the function assign_tissue to create the 'tissue' column
ALL.obs['tissue'] = ALL.obs['batch'].apply(assign_tissue)
print('\nCreated the "Tissue" column')

# Parameters to explore:
n_neighbors_list = [15, 30]
leiden_res = [0.5, 1.0, 1.5]

# Create a dictionary to store all the adatas
results = {}

# Perform clustering analysis looping on the parameters to explore, using our function
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