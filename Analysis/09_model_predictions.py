import joblib
import scanpy as sc
import numpy as np
import pandas as pd
from age_mapping import *


#### PREPARE THE DATA ####
# Load adata
adata = sc.read_h5ad('./aLL_cache/04_adata_n30_r1.0.h5ad')

# Load trained model
self_training_model = joblib.load('./ALL_cache/M1_best_classifier.pkl')

# Extract the embeddings
uce_embeddings = adata.obsm['X_uce']




#### PREDICT LABELS FOR ALL THE CELLS ####
# Apply the model
print("Predicting labels...")
predicted_labels = self_training_model.predict(uce_embeddings)
print("Prediction complete!")




#### ADD THE PREDICTED LABELS TO THE ADATA ####
# Convert numeric labels to string labels
label_mapping = {1: "senescent", 0: "non_senescent"}

# Add labels while transforming them to strings
adata.obs['predicted_labels'] = [label_mapping[label] for label in predicted_labels]
print('\nLabels added to the original adata :)')

# Save the cache
adata.write('./ALL_cache/09_adata_n30_r1.0.h5ad')
print('\nAdata successfully saved :)')




#### VISUALIZATION AND VALIDATION ####
# Counts and percentages of senescent cells:
label_counts = adata.obs['predicted_labels'].value_counts()
label_percentages = (label_counts / len(adata.obs)) * 100

print('RESULTS:')
print("Counts:\n", label_counts)
print("\nPercentages:\n", label_percentages)

# Group by tissue and predicted labels
counts_by_tissue = adata.obs.groupby(['tissue', 'predicted_labels']).size().unstack(fill_value=0)

# Calculate percentages
percentages_by_tissue = counts_by_tissue.div(counts_by_tissue.sum(axis=1), axis=0) * 100

print('RESULTS BY TISSUE:')
print("\nCounts by Tissue:\n", counts_by_tissue)
print("\nPercentages by Tissue:\n", percentages_by_tissue)


# PLOTS
# UMAP colored by senescent cells
sc.pl.umap(adata, color='predicted_labels', cmap='viridis', title='Predicted senescent cells', save='_predicted_SnCs.png')

# Plot the percentage of senescent cells according to age
adata.obs['age'] = adata.obs['batch'].map(age_mapping) # age-batch mapping
get_age_plots(adata, 'bone_marrow')


# DEA
# Differential expression analysis
print('\nDifferential expression analysis...')
sc.tl.rank_genes_groups(adata, groupby='predicted_labels', reference='non_senescent', method='wilcoxon')

# View top-ranked genes for the senescent group
top_genes = sc.get.rank_genes_groups_df(adata, group='predicted_labels')
print(top_genes)

sc.pl.rank_genes_groups_heatmap(adata, n_genes=10, groupby='predicted_labels', cmap='viridis', save='_DEA_SnCs_vs_nonSc.png')


# Save the cache
adata.write('./ALL_cache/09_adata_n30_r1.0.h5ad')
print('\nAdata successfully saved :)')