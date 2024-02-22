import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os
# path = "D:\\Project Work\\MAIN\\Boruta_20_feature\\9"
path = r"D:\Project Work\AFTER COMPRY MODEL REFINMENT\MODELS RESULTS\multitasking model\model_6899\fda_approved"
os.makedirs(path, exist_ok=True)

# Step 1: Load the X_train File
x_train_data = pd.read_csv(r"D:\Project Work\AFTER COMPRY MODEL REFINMENT\MODELS RESULTS\multitasking model\model_6899\X_train.csv") 

# Step 2: Load the Trained Model
model = load_model(r"D:\Project Work\AFTER COMPRY MODEL REFINMENT\multitasking model data and models\multitasking_model.h5")  

# Step 3: Load the CSV File for Prediction
prediction_data = pd.read_csv(r"D:\Project Work\AFTER COMPRY MODEL REFINMENT\FDA approve\main data\fda_used_for_pred.csv")  

# Step 4: Fill NaN values with 0
prediction_data.fillna(0, inplace=True)

# Step 5: Prepare the Data 
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_data)
print(x_train_scaled.shape)
prediction_data_scaled = scaler.transform(prediction_data)
print(prediction_data_scaled.shape)


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_data)
prediction_data_scaled = scaler.transform(prediction_data)

# Reshape to 2D if it has an extra dimension
# if len(prediction_data_scaled.shape) == 3:
#     prediction_data_scaled = prediction_data_scaled.squeeze(axis=-1)

# # Check for infinite values
# print("Number of infinite values in X_train:", not np.isfinite(x_train_scaled).all())
# print("Number of infinite values in prediction_data:", not np.isfinite(prediction_data_scaled).all())

# Step 6: Make Predictions
predictions = model.predict(prediction_data_scaled)


# Separate classification and regression outputs
classification_predictions = predictions[0]  
regression_predictions = predictions[1]  
print(type(regression_predictions))
np.savetxt(path+"/regression_fda_result.csv",regression_predictions , delimiter=",")

print("Classification Predictions:", classification_predictions)
# Convert classification predictions to labels
classification_labels = classification_predictions.round()
print(type(classification_labels))
np.savetxt(path+"/clasification_fda_result.csv",classification_labels , delimiter=",")

