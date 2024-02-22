import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from keras import regularizers
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score,accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, auc

# path = "D:\\Project Work\\MAIN\\Boruta_20_feature\\9"
path = "D:\\Project Work\\AFTER COMPRY MODEL REFINMENT\\MODELS RESULTS\\multitasking model\\model_6899\\results\\1"
os.makedirs(path, exist_ok=True)

file_path = r"D:\Project Work\AFTER COMPRY MODEL REFINMENT\multitasking model data and models\new_total_6899_data.csv"
df = pd.read_csv(file_path)
df.fillna(0)

X = df.iloc[:, :1923].values
y_classification = df.iloc[:, 1923].values
y_regression = df.iloc[:, 1924].values

X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_classification, y_regression, test_size=0.2)

# Convert arrays to DataFrames with feature names
df_X_train = pd.DataFrame(X_train, columns=df.columns[:1923].tolist())
df_X_test = pd.DataFrame(X_test, columns=df.columns[:1923].tolist())
df_y_class_train = pd.DataFrame({'y_classification': y_class_train})
df_y_class_test = pd.DataFrame({'y_classification': y_class_test})
df_y_reg_train = pd.DataFrame({'y_regression': y_reg_train})
df_y_reg_test = pd.DataFrame({'y_regression': y_reg_test})

# Save DataFrames to CSV files
df_X_train.to_csv(os.path.join(path, 'X_train.csv'), index=False)
df_X_test.to_csv(os.path.join(path, 'X_test.csv'), index=False)
df_y_class_train.to_csv(os.path.join(path, 'y_class_train.csv'), index=False)
df_y_class_test.to_csv(os.path.join(path, 'y_class_test.csv'), index=False)
df_y_reg_train.to_csv(os.path.join(path, 'y_reg_train.csv'), index=False)
df_y_reg_test.to_csv(os.path.join(path, 'y_reg_test.csv'), index=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load the model
model_path = r"D:\Project Work\AFTER COMPRY MODEL REFINMENT\multitasking model data and models\multitasking_model.h5"
loaded_model = load_model(model_path)

# Predict on the training data
y_train_pred_class, y_train_pred_reg = loaded_model.predict(X_train_scaled)

# Convert raw predictions to binary labels for Classification on Training Set
y_train_pred_class_labels = y_train_pred_class.round()

# Save binary predictions for Classification on Training Set
df_train_binary_predictions_class = pd.DataFrame({
    'True_Labels': y_class_train,
    'Binary_Predictions': y_train_pred_class_labels.flatten()
})
df_train_binary_predictions_class.to_csv(os.path.join(path, 'train_binary_predictions_classification.csv'), index=False)

# Save raw predictions for Regression on Training Set
df_train_raw_predictions_reg = pd.DataFrame({
    'Raw_Predictions': y_train_pred_reg[:, 0]  
})
df_train_raw_predictions_reg.to_csv(os.path.join(path, 'train_raw_predictions_regression.csv'), index=False)

# For Classification
y_train_pred_class_labels = y_train_pred_class.round()


conf_matrix_class = confusion_matrix(y_class_train, y_train_pred_class_labels)
classification_report_class = classification_report(y_class_train, y_train_pred_class_labels)

# Calculate Accuracy
accuracy = accuracy_score(y_class_train, y_train_pred_class_labels)

# Calculate Specificity and Sensitivity
tn, fp, fn, tp = conf_matrix_class.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

# Calculate AUROC
fpr, tpr, thresholds = roc_curve(y_class_train, y_train_pred_class)
auroc = auc(fpr, tpr)

# Calculate Positive Predictive Value (PPV) and Negative Predictive Value (NPV)
ppv = tp / (tp + fp)
npv = tn / (tn + fn)

# Print or use the metrics as needed

print("Classification Metrics:")
print("Confusion Matrix:\n", conf_matrix_class)
print("\nClassification Report:\n", classification_report_class)
print("Accuracy:", accuracy)
print("Specificity:", specificity)
print("Sensitivity:", sensitivity)
print("AUROC:", auroc)
print("Positive Predictive Value (PPV):", ppv)
print("Negative Predictive Value (NPV):", npv)

# For Regression
mse_reg = mean_squared_error(y_reg_train, y_train_pred_reg)
r2_reg = r2_score(y_reg_train, y_train_pred_reg)

# Print or use the metrics as needed
print("\nRegression Metrics:")
print("Mean Squared Error (MSE):", mse_reg)
print("R-squared (R2):", r2_reg)


# Predict on the test data
y_test_pred_class, y_test_pred_reg = loaded_model.predict(X_test_scaled)

# For Classification
y_test_pred_class_labels = y_test_pred_class.round()

# Save binary predictions for Classification on Test Set
df_test_binary_predictions_class = pd.DataFrame({
    'True_Labels': y_class_test,
    'Binary_Predictions': y_test_pred_class_labels.flatten()
})
df_test_binary_predictions_class.to_csv(os.path.join(path, 'test_binary_predictions_classification.csv'), index=False)
conf_matrix_test_class = confusion_matrix(y_class_test, y_test_pred_class_labels)

# Save raw predictions for Regression on Training Set
df_test_raw_predictions_reg = pd.DataFrame({
    'Raw_Predictions': y_train_pred_reg[:, 0]  
})
df_test_raw_predictions_reg.to_csv(os.path.join(path, 'test_raw_predictions_regression.csv'), index=False)

classification_report_test_class = classification_report(y_class_test, y_test_pred_class_labels)

# Calculate Accuracy on Test Set
accuracy_test = accuracy_score(y_class_test, y_test_pred_class_labels)

# Calculate Specificity and Sensitivity on Test Set
tn_test, fp_test, fn_test, tp_test = conf_matrix_test_class.ravel()
specificity_test = tn_test / (tn_test + fp_test)
sensitivity_test = tp_test / (tp_test + fn_test)

# Calculate AUROC on Test Set
fpr_test, tpr_test, thresholds_test = roc_curve(y_class_test, y_test_pred_class)
auroc_test = auc(fpr_test, tpr_test)

# Calculate Positive Predictive Value (PPV) and Negative Predictive Value (NPV) on Test Set
ppv_test = tp_test / (tp_test + fp_test)
npv_test = tn_test / (tn_test + fn_test)

# Print or use the metrics on the Test Set
print("Test Set Classification Metrics:")
print("Confusion Matrix:\n", conf_matrix_test_class)
print("\nClassification Report:\n", classification_report_test_class)
print("Accuracy:", accuracy_test)
print("Specificity:", specificity_test)
print("Sensitivity:", sensitivity_test)
print("AUROC:", auroc_test)
print("Positive Predictive Value (PPV):", ppv_test)
print("Negative Predictive Value (NPV):", npv_test)

# For Regression on Test Set
mse_reg_test = mean_squared_error(y_reg_test, y_test_pred_reg)
r2_reg_test = r2_score(y_reg_test, y_test_pred_reg)

# Print or use the metrics on the Test Set
print("\nTest Set Regression Metrics:")
print("Mean Squared Error (MSE):", mse_reg_test)
print("R-squared (R2):", r2_reg_test)

# Create a DataFrame to store evaluation metrics
metrics_data = {
    'Metric': ['Accuracy', 'Specificity', 'Sensitivity', 'AUROC', 'PPV', 'NPV', 'MSE', 'R2'],
    'Training': [accuracy, specificity, sensitivity, auroc, ppv, npv, mse_reg, r2_reg],
    'Test': [accuracy_test, specificity_test, sensitivity_test, auroc_test, ppv_test, npv_test, mse_reg_test, r2_reg_test]
}

metrics_df = pd.DataFrame(metrics_data)

# Save the DataFrame to a CSV file
metrics_file_path = os.path.join(path, 'evaluation_metrics.csv')
metrics_df.to_csv(metrics_file_path, index=False)

print("Evaluation metrics saved to:", metrics_file_path)

# Save Classification Report to a Text File
classification_report_file_path = os.path.join(path, 'classification_report.txt')
with open(classification_report_file_path, 'w') as report_file:
    report_file.write("Classification Report (Training Set):\n")
    report_file.write(classification_report_class)
    report_file.write("\n\n")

    report_file.write("Classification Report (Test Set):\n")
    report_file.write(classification_report_test_class)

print("Classification report saved to:", classification_report_file_path)

import matplotlib.pyplot as plt
# Plot R2 values for both training and test sets
plt.figure(figsize=(8, 5))
plt.plot(['Training', 'Test'], [r2_reg, r2_reg_test], marker='o')
plt.title('R-squared (R2) Comparison')
plt.xlabel('Dataset')
plt.ylabel('R-squared (R2) Value')
plt.grid(True)
plt.show()



#%%


print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% history data for plot% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


# Load the CSV file containing the model training history
history_path = r"D:\Project Work\AFTER COMPRY MODEL REFINMENT\multitasking model data and models\historyh5.csv"
history_df = pd.read_csv(history_path)

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# Plot Accuracy
plt.figure(figsize=(8, 6))
plt.plot(history_df['class_output_accuracy'], label='Training Accuracy')
plt.plot(history_df['val_class_output_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(path, 'accuracy_plot.png'))  

# Plot Loss
plt.figure(figsize=(8, 6))
plt.plot(history_df['loss'], label='Training Loss')
plt.plot(history_df['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(path, 'loss_plot.png'))  
plt.show()

def plot_precision_recall_curve(precision_train, recall_train, precision_test, recall_test, title):
    plt.figure(figsize=(8, 6))
    plt.plot(recall_train, precision_train, color='darkorange', lw=2, label='Train Precision-Recall curve')
    plt.plot(recall_test, precision_test, color='green', lw=2, label='Test Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(path, 'precision_recall_curve.png'))
    plt.show()

# Calculate Precision-Recall curve for the training set
precision_train, recall_train, _ = precision_recall_curve(y_class_train, y_train_pred_class)

# Calculate Precision-Recall curve for the test set
precision_test, recall_test, _ = precision_recall_curve(y_class_test, y_test_pred_class)

# Plot Precision-Recall curve
plot_precision_recall_curve(precision_train, recall_train, precision_test, recall_test, 'Precision-Recall Curve')

print("%%%%%%%%%%%%%%%%   ROC CURVE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ")

# Calculate ROC curve for the training set
fpr_train, tpr_train, _ = roc_curve(y_class_train, y_train_pred_class)

# Calculate ROC curve for the test set
fpr_test, tpr_test, _ = roc_curve(y_class_test, y_test_pred_class)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, color='darkorange', lw=2, label='Train ROC curve')
plt.plot(fpr_test, tpr_test, color='green', lw=2, label='Test ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig(os.path.join(path, 'roc_curve.png')) 
plt.show()