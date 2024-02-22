import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import tensorflow as tf

# Set the path
path = "D:/Project Work/MAIN/normaisation_NEW/4"
os.makedirs(path, exist_ok=True)

import joblib

# Load the scikit-learn model
model = joblib.load("D:\\Project Work\\MAIN\\Boruta_20_feature\\8\\best_random_forest_model.pkl")

# Load the training dataset
train_data = pd.read_csv("D:\\for gray_area_related\\train_set.csv")

# Extract the features from the training dataset
X_train = train_data  

scaler_train = StandardScaler()
X_train_normalized = scaler_train.fit_transform(X_train)

# Load the prediction dataset
# prediction_data = pd.read_csv("D:\\Project Work\\project work\\grayAREANEW.csv")
prediction_data = pd.read_csv("D:\\for gray_area_related\\x_train.csv")
# Use the same scaler that was fitted on the training data to normalize the prediction data
X_pred_normalized = scaler_train.transform(prediction_data)

# Make predictions on the prediction set
y_pred = model.predict(X_pred_normalized)

# Save the predictions
predictions_df = pd.DataFrame(y_pred, columns=["predicted_prob"])
predictions_df.to_csv(os.path.join(path, 'predictions.csv'), index=False)
