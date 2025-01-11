'''This file contains the functions required for the analysis across different files'''

import numpy as np
import os
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import gseapy as gp
import scvelo as scv
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


  
age_map = {
    'bm1_B1': 28,
    'bm1_B2': 46,
    'bm1_B3': 24,
    'bm1_B4': 55,
    'bm1_B5': 56,
    'bm1_B6': 55,
    'bm1_B7': 56,
    'bm1_B8': 31,
    'bm1_B9': 66,
    'bm1_B10': 58,
    'bm1_B11': 50,
    'bm1_B12': 67,
    'bm1_B13': 60,
    'bm1_B14': 57,
    'bm1_B15': 84,
    'bm1_B16': 43,
    'bm1_B17': 50,
    'bm1_B18': 58,
    'bm1_B19': 41,
    'bm1_B20': 30,
    'bm1_B21': 60,
    'bm1_B22': 59,
    'bm1_B23': 60,
    'bm1_B24': 47,
    'bm1_B25': 59,
    'sk1_S1': 29,
    'sk1_S2': 71,
    'sk1_S3': 70,
    'sk1_S4': 24,
    'sk1_S5': 67,
    'sk1_S6': 66,
    'sk1_S7': 24,
    'sk1_S8': 29
}

map_cell_types = {
  'ILC': 'ILC',
  'Monocytes': 'monocyte',
  'T cells': 'T_cell',
  'CD4 T cells': 'T_cell',
  'CD8 T cells': 'T_cell',
  'Treg': 'T_cell',
  'Th': 'T_cell',
  'Tc': 'T_cell',
  'Fibroblasts': 'fibroblast',
  'DC': 'dendritic_cell',
  'DC1': 'dendritic_cell',
  'DC2': 'dendritic_cell',
  'pDC': 'dendritic_cell',
  'moDC': 'dendritic_cell',
  'Myelocytes': 'myelocyte',
  'B cells': 'B_cell',
  'B-cell lineage': 'B_cell',
  'Plasma cells': 'plasma_cell',
  'Erythroid': 'erythroid',
  'MNP': 'monocyte_macrophage_progenitor',
  'Macrophages': 'macrophage',
  'Mono_mac': 'macrophage',
  'Inf_mac': 'macrophage',
  'Monocyte-derived Mph': 'macrophage',
  'Alveolar macrophages': 'alveolar_macrophage',
  'Alveolar Mph CCL3+': 'alveolar_macrophage',
  'Alveolar Mph MT-positive': 'alveolar_macrophage',
  'Interstitial Mph perivascular': 'macrophage',
  'HSC/MPP': 'hematopoietic_stem_cell',
  'Hematopoietic stem cells': 'hematopoietic_stem_cell',
  'Megakaryocytes/platelets': 'megakaryocyte_platelet',
  'Megakaryocyte precursor': 'megakaryocyte_platelet',
  'Promyelocytes': 'promyelocyte',
  'Mast cells': 'mast_cell',
  'Differentiated_KC': 'differentiated_KC',
  'Undifferentiated_KC': 'undifferentiated_KC',
  'Melanocyte': 'melanocyte',
  'migLC': 'langerhans_cell',
  'LC': 'langerhans_cell',
  'ILC1_NK': 'ILC',
  'ILC1_3': 'ILC',
  'Endothelial cells': 'endothelial_cell',
  'Lymphatic EC mature': 'endothelial_cell',
  'EC arterial': 'endothelial_cell',
  'EC venous pulmonary': 'endothelial_cell',
  'EC general capillary': 'endothelial_cell',
  'Alveolar fibroblasts': 'fibroblast',
  'Adventitial fibroblasts': 'fibroblast',
  'Subpleural fibroblasts': 'fibroblast',
  'Pericytes': 'pericyte',
  'Pericyte_1': 'pericyte',
  'Pericyte_2': 'pericyte',
  'Non-classical monocytes': 'monocyte',
  'Classical monocytes': 'monocyte',
  'Monocyte precursor': 'monocyte',
  'NK cells': 'NK_cell',
  'Smooth muscle': 'smooth_muscle',
  'Mesothelium': 'mesothelium',
  'Multiciliated (non-nasal)': 'ciliated_cell',
  'AT1': 'alveolar_epithelial_cell',
  'AT2': 'alveolar_epithelial_cell',
  'Macro_1': 'macrophage',
  'Macro_2': 'macrophage',
  'Suprabasal': 'Suprabasal',
  'pre-TB secretory': 'secretory_cell',
  'VE1': 'vascular_endothelium',
  'VE2': 'vascular_endothelium',
  'VE3': 'vascular_endothelium',
  'F2': 'F2',
  'LE1': 'LE1',
  'Alveolar Mph proliferating': 'alveolar_macrophage'
}

#### ASSIGN TISSUE LABELS TO ADATA ####
def assign_tissue(batch):
  '''Function to label the cells based on their tissue origin (batch)'''
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


#### CLUSTERING ####
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


#### PLOT SnCs vs AGE ####
def get_age_plots(adata, tissue, sen_labels='Sen_consensus'):
  '''This function plots the percentage of senescent cells by age in a given tissue'''
  
  import matplotlib.pyplot as plt
  import seaborn as sns
  import pandas as pd

  # Drop cells with NaN ages and select only bone marrow samples
  adata_filtered = adata[adata.obs['tissue'] == tissue].copy()
  adata_filtered = adata_filtered[~adata_filtered.obs['age'].isna()].copy()

  # Group by age and sen_labels, then count
  age_distribution = adata_filtered.obs.groupby(['age', sen_labels], observed=False).size().reset_index(name='count')

  # Calculate total cells for each age group
  total_cells_per_age = age_distribution.groupby('age')['count'].transform('sum')

  # Calculate percentages of senescent cells for each age
  age_distribution['percentage'] = (age_distribution['count'] / total_cells_per_age) * 100

  # Bar plot: percentage of senescent cells vs age
  senescent_data = age_distribution[age_distribution[sen_labels] == 'senescent']

  # Bar plot for senescent cells percentage
  plt.figure(figsize=(10, 6))
  sns.barplot(data=senescent_data, x='age', y='percentage', color='red') 
  plt.title(f'Percentage of Senescent Cells by Age in {tissue}')
  plt.xlabel('Age')
  plt.ylabel('Percentage of Senescent Cells')
  plt.tight_layout() 
  plt.show()


#### DOTPLOT DEAs ####
def DEA_dotplot(adata, degs_df_dic, groupby, n, png_name):
  '''This function creates a dotplot comparing the top n DEGs for each group.
  degs_df_dic = a dictionary of DEGs, one entry for each group.
  groupby = the groups label in the adata'''
  
  gene_set = set() # A set to store the selected genes for each group (it doesn´t store duplicates)
  ngenes = n # number of top genes to extract from each result

  for df in degs_df_dic.values():
    top_genes = set(df.head(ngenes)["names"].values) # select the top n genes
    gene_set.update(top_genes) # add the genes to the set

  print('\nPreparing the dotplot...')
  fig, ax = plt.subplots(1,1, figsize=(5, 7))
  sc.pl.dotplot(adata, groupby=groupby, var_names=list(gene_set), show=True, swap_axes=True, ax=ax, save=png_name)


#### GSEA ####
def compute_gsea(degs_df, outdir, top_n=2000):
  '''
  This function computes a pre-ranked GSEA using log fold change as the ranking metric.
  The input is a dataframe with DEGs, gene names and log fold change values. The analysis 
  is done only on the top N genes with significant adjusted p-values.
  '''
  print("\nComputing Pre-ranked GSEA...")

  # Filter significant genes and sort by logfoldchanges
  filtered_genes = degs_df[degs_df['pvals_adj'] < 0.05].sort_values(by='logfoldchanges', ascending=False).head(top_n)

  # Create a ranked list: gene names with their logfoldchanges
  ranked_df = filtered_genes[['names', 'logfoldchanges']].dropna()

  # Perform Pre-ranked GSEA
  gsea = gp.prerank(
      rnk=ranked_df,
      gene_sets='GO_Biological_Process_2021',
      permutation_num=100,  # Number of permutations
      outdir=outdir,
      seed=6
  )
  
  #### CREATE A READABLE EXCEL WITH THE OUTPUT ####
  # # Load the CSV file
  csv_file = f"{outdir}/gseapy.gene_set.prerank.report.csv"  
  data = pd.read_csv(csv_file)

  # Select only the columns of interest: GO term, FDR q-value, and NES
  columns = ["Term", "FDR q-val", "NES"]
  filtered_data = data[columns]

  # Save the filtered data to an Excel file
  excel_file = f"{outdir}/filtered_GO_terms.xlsx"
  filtered_data.to_excel(excel_file, index=False)

  print(f"Excel file saved in {excel_file}")

  return gsea


#### ORA ####
def compute_ora(degs_df, out_excel, out_IDs):
  '''This function computes an ORA. 
  degs_df = dataframe with the DEGs for that group.
  It outputs an excel file with the ORA results as well as an excel file with 
  the filtered GO-IDs and p-values ready for Revigo analysis'''

  print("\nComputing ORA...")

  # Perform analysis using GO Biological Process
  significant_genes = degs_df[degs_df['pvals_adj'] < 0.05]['names'] # extract only the gene names

  enr = gp.enrichr(
      gene_list=significant_genes,
      gene_sets=['GO_Biological_Process_2021'], 
      organism='Human',
      cutoff=0.05)
  print("\nenrichr done!")

  # Save results to an excel
  enr.results.to_excel(out_excel, index=False)
  print(f"\nGSEA results saved to excel {out_excel}")

  # Extract the GO IDs and Adjusted P-values to another excel file (input for Revigo tool)
  # Read the Excel file generated previously
  data = pd.read_excel(out_excel)

  # Extract GO IDs and Adjusted P-values
  data['GO_ID'] = data['Term'].str.extract(r'\((GO:\d+)\)')
  result = data[['GO_ID', 'Adjusted P-value']]

  # Save the result to a new Excel file
  result.to_excel(out_IDs, index=False)

  print(f"Filtered GO IDs saved to {out_IDs}")


#### CALCULATE SIGNATURE SCORES ####
def calculate_signature_score(adata, up_genes, down_genes=None, signature="signature_name"):
  '''This function applies a signature to calculate the score of upregulated genes and substracts the score of downreguated genes'''
  
  sc.tl.score_genes(adata, gene_list=up_genes, score_name=signature) # score for upregulated genes
  
  if down_genes: # If downregulated genes are provided, substract this score
    sc.tl.score_genes(adata, gene_list=down_genes, score_name=f'{signature}_down')
    adata.obs[signature] = adata.obs[signature] - adata.obs[f'{signature}_down']
  return adata


#### LABEL ARRESTED CELLS IN ADATA ####
def label_arrested_cells(adata, threshold = 0.2):
  '''This function calculates cell cycle scores for each cell and labels them as arrested (TRUE) or non arrested (FALSE)
  based on a given score threshold (default=0.2)'''

  # Score cell cycle phases
  scv.tl.score_genes_cell_cycle(adata)
  ''' Calculates scores and assigns a cell cycle phase (G1, S, G2M) using the list of cell cycle genes defined in [Tirosh et al., 2016].'''

  # Labelling arrested cells
  ''' Arrested cells have low G1/S and G2/M scores'''
  adata.obs["cell_cycle_arrest"] = ((adata.obs["S_score"] < threshold) & (adata.obs["G2M_score"] < threshold)) # TRUE = arrested

  return adata


#### FUNCTIONS FOR RANDOM FOREST MODEL ####
def label_extraction(adata):
  '''This function extracts and prepares the senescent labels from the adata. It returns the following items:
  
  adata = updated adata with senescent labels in numeric values stored in .obs["Sen_core_numeric"]
  x = a numpy array with UCE embeddings;
  y = a numpy array with senescent labels in numeric values: "senescent": 1, "non_senescent": 0, "unknown": -1'''

  print('\nPreparing the data...')
  # Convert senescent labels to numeric values
  label_mapping = {'senescent': 1, 'non_senescent': 0, 'unknown': -1}
  adata.obs['Sen_core_numeric'] = adata.obs['Sen_core'].map(label_mapping)

  # Extract embeddings and labels
  x = adata.obsm['X_uce'] 
  y = adata.obs['Sen_core_numeric'].values

  return adata, x, y


def downsample(x, y, size=40000):
  '''This function downsamples the x and y arrays, retaining all the senescent cells if less than 20%, but reducing the 
  amount of non_senescent and unknown cells, to have a more balanced proportion and a managable total size.
  
  It returns:
  x_subset (numpy array): a subset of x. 
  y_subset (numpy array): a subset of y. 
  
  '''
  
  print('\nDownsampling the dataset...')
  # Set the seed for reproducibility
  np.random.seed(47360368)

  # Get indices for each class
  sen_indices = np.where(y == 1)[0] # Returns the indices for all the cells labeled as senescent
  non_sen_indices = np.where(y == 0)[0] # Returns the indices for all the cells labeled as non_senescent
  unknown_indices = np.where(y == -1)[0] # Returns the indices for all the cells labeled unknown

  # Define target sizes
  N = size  # Total target size for the training dataset. Default = 40000
  sen_size = min(len(sen_indices), int(0.2 * N))  # full length of senescent cells if less than 20% of N
  non_sen_size = int(0.50 * N)  # 50% non-senescent
  unknown_size = N - sen_size - non_sen_size  # Remaining for unknown

  # Generate random samples for each category 
  sen_sample = np.random.choice(sen_indices, size=sen_size, replace=False) 
  non_sen_sample = np.random.choice(non_sen_indices, size=non_sen_size, replace=False)
  unknown_sample = np.random.choice(unknown_indices, size=unknown_size, replace=False)

  # Combine samples
  subset_indices = np.concatenate([sen_sample, non_sen_sample, unknown_sample])

  # Subset the data based on the generated samples
  x_subset = x[subset_indices]
  y_subset = y[subset_indices]

  return x_subset, y_subset


def get_train_val_sets(x_subset, y_subset, split=0.2):
  '''This function splits the dada into training and validation sets. The default split is set
  to 80/20: 80% training set and 20% validation set. The validation set will contain only labeled data. 

  Outputs:
  x_train, y_train (numpy arrays): Training set (80% of labeled data + 100% unlabeled data).
  x_val, y_val: Validation set (20% of labeled data).

  '''

  print('\nCreating training and validation sets...')
  # Separate the labeled cells (0 or 1) and the unlabeled cells (-1)
  labeled_indices = np.where(y_subset != -1)[0] # Returns the indices for all the cells labeled as senescent or non_senescent
  unlabeled_indices = np.where(y_subset == -1)[0] # Returns the indices for all the unknown cells 

  # Split the labeled cells into 80/20 datasets.
  train_indices, val_indices = train_test_split(labeled_indices, test_size=split, random_state=47, stratify=y_subset[labeled_indices])
  '''
  test_size : proportion of sample used as validation set. Default=0.2
  random_state=50 : sets a fixed random seed for reproducibility
  stratify : ensures both sets (training and validation) retain the same proportions for each class, equal to the original set'''

  # Add the unlabeled cells to the training set
  train_indices = np.concatenate([train_indices, unlabeled_indices])

  # Final training sets and validation sets
  train_y = np.copy(y_subset[train_indices])
  train_x = np.copy(x_subset[train_indices])

  val_y = np.copy(y_subset[val_indices])
  val_x = np.copy(x_subset[val_indices])

  return train_x, train_y, val_x, val_y

