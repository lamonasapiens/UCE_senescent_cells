'''This code tests the parameters for the Self Training Classifier'''

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from rf_model_setup import *
import numpy as np
import scanpy as sc
import joblib
import pandas as pd



#### LOAD DATA ####
adata = sc.read_h5ad('./aLL_cache/04_adata_n30_r1.0.h5ad')

#### PREPARE THE DATA ####
adata, x, y = label_extraction(adata)

#### DOWNSAMPLING THE DATA ####
x_subset, y_subset = downsample(x, y, size=10000) # use a smaller sample to speed up the process

#### DIVIDE DATA IN 80/20 ####
train_x, train_y, val_x, val_y = get_train_val_sets(x_subset, y_subset, split=0.2)



#### FINE TUNNING WORKFLOW 2 ####
print('\nStarting the fine-tuning process...')

param_grid = {
    'k_best': [10, 20],
    'max_iter': [5, 10, 20]
}

# Base classifier
# Initialize random forest classifier (with the best parameters found in the previous step)
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=50,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',  
    random_state=42           
)

# Store results
results = []

# Iterate through the parameters
for k_best in param_grid['k_best']:
  for max_iter in param_grid['max_iter']:
    print(f"Testing k_best={k_best}, max_iter={max_iter}...")
    
    # Initialize Self-TrainingClassifier with current parameters
    model = SelfTrainingClassifier(rf_model, criterion="k_best", k_best=k_best, max_iter=max_iter)
    
    # Fit on labeled data
    model.fit(train_x, train_y)
    
    # Predict on validation set
    val_predictions = model.predict(val_x)
    
    # Evaluate performance
    report = classification_report(val_y, val_predictions, output_dict=True)
    results.append({
      'k_best': k_best,
      'max_iter': max_iter,
      'precision': report['weighted avg']['precision'],
      'recall': report['weighted avg']['recall'],
      'f1-score': report['weighted avg']['f1-score']
    })
print('\nTesting finished!')


# Convert results to a DataFrame for easy analysis
print('\nPreparing the results...')
results_df = pd.DataFrame(results)
results_df.sort_values(by='f1-score', ascending=False, inplace=True)

# Display results
results_df.to_csv("self_training_results.csv", index=False)
print("Results saved to 'self_training_results.csv'")