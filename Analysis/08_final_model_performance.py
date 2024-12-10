
import numpy as np
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from rf_model_setup import *


#### LOAD DATA ####
adata = sc.read_h5ad('./aLL_cache/04_adata_n30_r1.0.h5ad')

#### PREPARE THE DATA ####
adata, x, y = label_extraction(adata)

#### DOWNSAMPLING THE DATA ####
x_subset, y_subset = downsample(x, y, size=50000)

#### DIVIDE DATA IN 80/20 ####
train_x, train_y, val_x, val_y = get_train_val_sets(x_subset, y_subset, split=0.2)



#### TRAIN THE OPTIMIZED MODEL ####

# Initialize random forest classifier
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=50,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',  # Handles class imbalance
    random_state=42           # Ensures reproducibility
)

# Initialize self training classifier
self_training_model = SelfTrainingClassifier(base_estimator=rf_model, criterion="k_best", k_best=20, max_iter=10)

# Fit on labeled data
print('\nTraining the model...')
self_training_model.fit(train_x, train_y)
print('Training done!')

# Evaluate on the validation set
print('\nComputing predictions...')
predictions = self_training_model.predict(val_x)



#### EVALUATION OF MODEL PERFORMANCE ####
print('\nRESULTS:')
print('\nAccuracy score:')
print(accuracy_score(val_y, predictions))
print('\nReport')
print(classification_report(val_y, predictions, target_names=["Non_senescent", "Senescent"]))
print('\nConfusion matrix')
print(confusion_matrix(val_y, predictions))



#### SAVE INTO CACHE ####
import joblib

# Defining cache file paths
model_cache_path = "./ALL_cache/M1_best_classifier.pkl"
predictions_cache_path = "./ALL_cache/M1_predictions.npy"

print("\nSaving model and predictions to cache...")
joblib.dump(self_training_model, model_cache_path)
joblib.dump(predictions, predictions_cache_path)
print("Model and predictions saved!")