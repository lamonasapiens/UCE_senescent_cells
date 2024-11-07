import scanpy as sc

# Load the embeddings
ALL = sc.read_h5ad('../Datasets/UCE_output/ALL/ALL_adata_uce_adata.h5ad')
print(ALL) # With this print we can see that the label X_uce has been generated as expected.

# Pull out the embeddings
uce = ALL.obsm["X_uce"]
print(uce.shape) # Here we see that our embeddings contain 1,280 dimentions for 108,861 cells

# show the metadata
print(ALL.obs)



# CLUSTERING
# First we need to construct a nearest-neighbours graph. By default, scanpy uses the Euclidean distance to find the nearest neighbors
sc.pp.neighbors(ALL, use_rep = "X_uce") 

# Perform the clustering with Leiden algorithm
sc.tl.leiden(ALL, resolution=1.0) 
# After running sc.tl.leiden(), the identified cluster labels are added to sk1.obs['leiden']. Each cell is assigned a cluster label

# Run umap to visualize clusters
sc.tl.umap(ALL)
sc.pl.umap(ALL, color=['leiden']) # colour by cluster
sc.pl.umap(ALL, color=['batch']) # colour by dataset