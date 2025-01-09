'''This code trains a semi-supervised learning model using Random Forest that will learn to 
classify senescent and non-senescent cells. '''

import numpy as np
import scanpy as sc
import joblib
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from funciones import * 


#### LOAD DATA ####
adata = sc.read_h5ad('./aLL_cache/08_adata_n30_r1.0.h5ad')

#### PREPARE THE DATA ####
adata, x, y = label_extraction(adata) # function from funciones.py

#### DOWNSAMPLING THE DATA ####
x_subset, y_subset = downsample(x, y, size=40000)  # function from funciones.py

#### DIVIDE DATA IN 80/20 ####
train_x, train_y, val_x, val_y = get_train_val_sets(x, y, split=0.2)  # function from funciones.py



#### TRAINING THE MODEL ####
# Initialize the base classifier (e.g., Random Forest)
base_classifier = RandomForestClassifier(n_jobs=-1, random_state=47)

# Initialize the Self-Training model
self_training_model = SelfTrainingClassifier(base_classifier, criterion='k_best', k_best=20, max_iter=10)  

# Training the model
print('\nTraining the model...')
self_training_model.fit(train_x, train_y)
print('Model successfully trained!')

# Predict labels for the validation set
print('\nPredicting labels for the validation set...')
predictions = self_training_model.predict(val_x)



#### EVALUATION OF MODEL PERFORMANCE ####
print('\nRESULTS:')
print('\nAccuracy score:')
print(accuracy_score(val_y, predictions))
print('\nReport')
print(classification_report(val_y, predictions, target_names=["Non_senescent", "Senescent"]))
print('\nConfusion matrix')
print(confusion_matrix(val_y, predictions))



#### SAVE THE MODEL ####
print("\nSaving model to cache...")
joblib.dump(self_training_model, "./ALL_cache/Model_1.pkl")
print("Model saved!")