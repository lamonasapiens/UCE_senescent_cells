'''This code trains a semi-supervised learning model using Random Forest that will learn to 
classify senescent and non-senescent cells. How it works:

The imput consists of two datasets: x (features, e.g. UCE embeddings) and y (labels). The labels can be the following:
- senescent -> 1
- non_senescent -> 0
- unknown -> -1 (considered unlabelled cells)

Due to computational constrains as well as a very low representation of the senescent cells category,
I downsampled the non_senescent cells, aiming to achieve a N= 50,000 and a balanced representation
for each category:

- senescent -> 13%
- non_senescent -> 55%
- unknown -> 32%
 '''

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
adata, x, y = label_extraction(adata) # function from rf_model_setup

#### DOWNSAMPLING THE DATA ####
x_subset, y_subset = downsample(x, y, size=50000)  # function from rf_model_setup

#### DIVIDE DATA IN 80/20 ####
train_x, train_y, val_x, val_y = get_train_val_sets(x_subset, y_subset, split=0.2)  # function from rf_model_setup



#### TRAINING THE MODEL ####
# Initialize the base classifier (e.g., Random Forest)
base_classifier = RandomForestClassifier(class_weight="balanced", random_state=42)

# Initialize the Self-Training model
self_training_model = SelfTrainingClassifier(base_classifier, criterion="k_best", k_best=20, max_iter=10)  

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

