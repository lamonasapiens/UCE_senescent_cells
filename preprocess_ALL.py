import scanpy as sc
import anndata as ad
import pandas as pd

# LOADING THE DATA
# List of the paths for all the sk1 and bm1 samples
paths = [f'../Datasets/sk1/S{i}' for i in range(1, 9)] + [f'./bm1/B{i}' for i in range(1, 9)]

# Load the data into AnnData objects
adata_list = []
for path in paths:
    adata = sc.read_10x_mtx(path, var_names='gene_symbols', cache=True)
    adata.var_names_make_unique()  # make variables unique
    adata_list.append(adata)

# Load Lu1 data
adata_lu1 = sc.read_10x_h5("./lu1/GSM4618899_SC340raw_feature_bc_matrix.h5")
adata_lu1.var_names_make_unique() # make variables unique
adata_list.append(adata_lu1)

# Load sk3 data
adata_sk3 = sc.read_10x_h5("./sk3/filtered_feature_bc_matrix.h5")
adata_sk3.var_names_make_unique() # make variables unique
adata_list.append(adata_sk3)
  
# Concatenate all the objects into one anndata
labels =[f'sk1_S{i}' for i in range(1, 9)] + [f'bm1_B{i}' for i in range(1, 9)] + ['lu1', 'sk3']
adata = sc.concat(adata_list, axis=0, join='outer', merge='first', label='batch', keys=labels, index_unique='-')




# FILTERING AND METRICS
# Check the number of cells and genes
print('\nRaw data:')
print(f"Number of cells: {adata.n_obs}")
print(f"Number of genes: {adata.n_vars}")

# Calculate quality control (QC) metrics
sc.pp.calculate_qc_metrics(adata, inplace=True)

# Filter low quality cells and genes
sc.pp.filter_cells(adata, min_genes=250)
sc.pp.filter_genes(adata, min_cells=5)

# Check the number of cells and genes after filtering
print('\nAfter filtering:')
print(f"Number of cells: {adata.n_obs}")
print(f"Number of genes: {adata.n_vars}")

# Print stats per batch
print("\nCells per batch:")
print(adata.obs['batch'].value_counts())



# SAVE THE DATA
# Save the adata into a .h5ad file
adata.write('./preprocessed_adatas/ALL_adata.h5ad')
