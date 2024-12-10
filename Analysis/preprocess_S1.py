import scanpy as sc
import anndata as ad
import pandas as pd

# Load S1
adata_S1 = sc.read_10x_mtx('./sk1/S1', var_names='gene_symbols', cache=True)

# Check the number of cells and genes before filtering
print(f"Number of cells: {adata_S1.n_obs}")
print(f"Number of genes: {adata_S1.n_vars}")

# Filter low quality cells and genes
sc.pp.filter_cells(adata_S1, min_genes=200)
sc.pp.filter_genes(adata_S1, min_cells=5)

# Check the number of cells and genes after filtering
print(f"Number of cells: {adata_S1.n_obs}")
print(f"Number of genes: {adata_S1.n_vars}")

# Save the S1 adata into a .h5ad file
adata_S1.write('./preprocessed_adatas/S1_adata.h5ad')