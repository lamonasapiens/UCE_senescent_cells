'''This code fine-tunes the hyperparameters of the Random Forest Base Classifier, giving the best
combination that improves the F1 scores and saving the best model.'''

import numpy as np
import scanpy as sc
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from funciones import *

#### LOAD DATA ####
adata = sc.read_h5ad('./aLL_cache/05_adata_n30_r1.0.h5ad')

#### PREPARE THE DATA ####
adata, x, y = label_extraction(adata)

#### DOWNSAMPLING THE DATA ####
x_subset, y_subset = downsample(x, y, size=40000)

#### DIVIDE DATA IN 80/20 ####
train_x, train_y, val_x, val_y = get_train_val_sets(x_subset, y_subset, split=0.2)



#### FINE-TUNING WORKFLOW ####
print('\nStarting the fine-tuning process...')

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [20, 50],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'max_features': ['sqrt'],
}

# Base Random Forest model
rf = RandomForestClassifier(n_jobs=-1, random_state=47)

# Initialize the Grid search with cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='f1_weighted', verbose=2, n_jobs=-1)
'''GridSearchCV is a tool that automates the process of tuning hyperparameters for a given estimator. 
It finds the combination of parameters that produce the best performance based on a chosen scoring metric.

cd = specifies the number of cross-validation folds to use (here it´s set to 3 splits
n_jobs=-1 means all available CPU cores will be utilized, speeding up the grid search
'''

# Start the searching process (training)
labeled_indices = train_y != -1  # The base classifier needs only labeled data
grid_search.fit(train_x[labeled_indices], train_y[labeled_indices])
print('Best parameters found!')

# Output best parameters
print("\nBest parameters:", grid_search.best_params_)




#### FIT THE BEST MODEL ####
# Initialize the optimized Random Forest
optimized_rf = grid_search.best_estimator_

# Initialize Self-Training model
self_training_model = SelfTrainingClassifier(base_estimator=optimized_rf, criterion="k_best", k_best=20, max_iter=10)

# Train the model
print('\nTraining the best model...')
self_training_model.fit(train_x, train_y)
print('Training done!')

# Predict labels on the validation set
print('\nComputing predictions...')
predictions = self_training_model.predict(val_x)



#### EVALUATION OF MODEL PERFORMANCE ####
print('\nRESULTS:')
print('\nAccuracy score:')
print(accuracy_score(val_y, predictions))
print('\nReport')
print(classification_report(val_y, predictions, target_names=["Non-senescent", "Senescent"]))
print('\nConfusion matrix')
print(confusion_matrix(val_y, predictions))



#### SAVE THE MODEL ####
print("\nSaving model to cache...")
joblib.dump(self_training_model, "./ALL_cache/Model_best.pkl")
print("Model saved!")
