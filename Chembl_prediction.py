
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix,mean_squared_error, r2_score
from tensorflow.keras.models import load_model
import os

# Load prediction data
prediction_file_path = r"D:\Project Work\AFTER COMPRY MODEL REFINMENT\Chemble\chemble_extacted_data.csv"
df = pd.read_csv(prediction_file_path)
df.fillna(0)

X_pred = df.iloc[:, :1923].values
y_classification = df.iloc[:, 1923].values
y_regression = df.iloc[:, 1924].values

# Load the model
model_path = r"D:\Project Work\AFTER COMPRY MODEL REFINMENT\multitasking model data and models\multitasking_model.h5"

# model 1 "D:\Project Work\AFTER COMPRY MODEL REFINMENT\multitasking model data and models\multitasking_model.h5"

loaded_model = load_model(model_path)

# Load the training data (assuming 'X_train_csv' is your training data loaded from CSV)
X_train_csv = pd.read_csv(r"D:\Project Work\AFTER COMPRY MODEL REFINMENT\MODELS RESULTS\multitasking model\model_6899\X_train.csv")

# Fit a new scaler on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_csv)

# Normalize prediction data
X_pred_normalized = scaler.transform(X_pred)

# Make predictions on the normalized prediction data
predict_class, predict_reg = loaded_model.predict(X_pred_normalized)

# Convert predictions to NumPy arrays
y_train_pred_class = predict_class.round()
y_train_pred_regression = predict_reg.flatten()  # Assuming predict_reg is a 1D array

# Save binary predictions for Classification on Training Set
df_predictions_class = pd.DataFrame({
    'True_Labels': y_classification,
    'Binary_Predictions': y_train_pred_class.flatten()
})
path_class = "D:\Project Work\AFTER COMPRY MODEL REFINMENT\MODELS RESULTS\multitasking model\model_6899\chemble"
os.makedirs(path_class, exist_ok=True)
df_predictions_class.to_csv(os.path.join(path_class, 'predictions_classification.csv'), index=False)

# Save raw predictions for Regression on Training Set
df_predictions_reg = pd.DataFrame({
    'Raw_Predictions': y_train_pred_regression
})
path_reg = "D:\Project Work\AFTER COMPRY MODEL REFINMENT\MODELS RESULTS\multitasking model\model_6899\chemble"
os.makedirs(path_reg, exist_ok=True)
df_predictions_reg.to_csv(os.path.join(path_reg, 'predictions_regression.csv'), index=False)

# Evaluate the performance for Classification
print("Classification Report:")
print(classification_report(y_classification, y_train_pred_class))
print("Confusion Matrix:")
print(confusion_matrix(y_classification, y_train_pred_class))
print("AUROC for Classification:", roc_auc_score(y_classification, y_train_pred_class))

# Evaluate the performance for Regression
print("\nRegression Metrics:")
print("Mean Squared Error (MSE):", mean_squared_error(y_regression, y_train_pred_regression))
print("R-squared (R2):", r2_score(y_regression, y_train_pred_regression))