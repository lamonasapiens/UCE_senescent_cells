'''This file contains the functions required to prepare the data for training the Semi-supervised Random Forest model,
used in files nº 05, 06 and 07'''

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import scanpy as sc

def label_extraction(adata):
  '''This function extracts and prepares the senescent labels from the adata. It returns the following items:
  
  adata = updated adata with senescent labels in numeric values stored in .obs["Sen_consensus_numeric"]
  x = a numpy array with UCE embeddings;
  y = a numpy array with senescent labels in numeric values: "senescent": 1, "non_senescent": 0, "unknown": -1'''

  print('\nPreparing the data...')
  # Convert senescent labels to numeric values
  label_mapping = {'senescent': 1, 'non_senescent': 0, 'unknown': -1}
  adata.obs['Sen_consensus_numeric'] = adata.obs['Sen_consensus'].map(label_mapping)

  # Extract embeddings and labels
  x = adata.obsm['X_uce'] 
  y = adata.obs['Sen_consensus_numeric'].values

  # Standardize the UCE embeddings (so that each dimension has the same scale)
  scaler = StandardScaler()
  x = scaler.fit_transform(x)

  return adata, x, y

def downsample(x, y, size=50000):
  '''This function downsamples the x and y arrays, retaining all the senescent cells but reducing the 
  amount of non_senescent and unknown cells, to have a more balanced proportion and a managable total size.
  
  It returns:
  x_subset (numpy array): a subset of x. 
  y_subset (numpy array): a subset of y. 
  
  '''
  
  print('\nDownsampling the dataset...')
  # Get indices for each class
  sen_indices = np.where(y == 1)[0] # Returns the indices for all the cells labeled as senescent
  non_sen_indices = np.where(y == 0)[0] # Returns the indices for all the cells labeled as non_senescent
  unknown_indices = np.where(y == -1)[0] # Returns the indices for all the cells labeled unknown

  # Define target sizes
  N = size  # Total target size for the training dataset. Default = 50000
  sen_size = min(len(sen_indices), int(0.2 * N))  # full length of senescent cells if less than 20% of N
  non_sen_size = int(0.55 * N)  # 55% non-senescent
  unknown_size = N - sen_size - non_sen_size  # Remaining for unknown

  # Generate random samples for each category (except for senescent, which will be used fully)
  sen_sample = np.random.choice(sen_indices, size=non_sen_size, replace=False) 
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
  train_indices, val_indices = train_test_split(labeled_indices, test_size=split, random_state=42, stratify=y_subset[labeled_indices])
  '''
  test_size : proportion of sample used as validation set. Default=0.2
  random_state=42 : sets a fixed random seed for reproducibility
  stratify : ensures both sets (training and validation) retain the same proportions for each class, equal to the original set'''

  # Add the unlabeled cells to the training set
  train_indices = np.concatenate([train_indices, unlabeled_indices])

  # Final training sets and validation sets
  train_y = np.copy(y_subset[train_indices])
  train_x = np.copy(x_subset[train_indices])

  val_y = np.copy(y_subset[val_indices])
  val_x = np.copy(x_subset[val_indices])

  return train_x, train_y, val_x, val_y