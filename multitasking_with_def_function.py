import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score, accuracy_score, roc_curve, auc

# Function to calculate classification metrics
def calculate_classification_metrics(y_true, y_pred, prefix=""):
    labels = y_pred.round()
    conf_matrix = confusion_matrix(y_true, labels)
    classification_rep = classification_report(y_true, labels, output_dict=True)
    accuracy = accuracy_score(y_true, labels)

    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auroc = auc(fpr, tpr)

    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    print(f"{prefix}Classification Metrics:")
    print("Confusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", classification_rep)
    print("Accuracy:", accuracy)
    print("Specificity:", specificity)
    print("Sensitivity:", sensitivity)
    print("AUROC:", auroc)
    print("Positive Predictive Value (PPV):", ppv)
    print("Negative Predictive Value (NPV):", npv)

    return conf_matrix, classification_rep, accuracy, specificity, sensitivity, auroc, ppv, npv

# Function to calculate regression metrics
def calculate_regression_metrics(y_true, y_pred, prefix=""):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"{prefix}Regression Metrics:")
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)

    return mse, r2

# Save CSV files
def save_to_csv(data, filename, prefix=""):
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(path, f'{prefix.lower()}{filename}.csv'), index=False)

# path = "D:\\Project Work\\MAIN\\Boruta_20_feature\\9"
path = "D:\\Project Work\\AFTER COMPRY MODEL REFINMENT\\MODELS RESULTS\\multitasking model/1"
os.makedirs(path, exist_ok=True)

file_path = "D:\\Project Work\\AFTER COMPRY MODEL REFINMENT\\working_data_nitin_new.csv"
df = pd.read_csv(file_path)
df.fillna(0)

X = df.iloc[:, :1923].values
y_classification = df.iloc[:, 1923].values
y_regression = df.iloc[:, 1924].values

X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_classification, y_regression, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load the model
model_path = "C:\\Users\\wankh\\Downloads\\multitasking_model.h5"
loaded_model = load_model(model_path)

# Predict on the training data
y_train_pred_class, y_train_pred_reg = loaded_model.predict(X_train_scaled)

# Calculate and print classification metrics for training set
train_conf_matrix, train_classification_rep, train_accuracy, train_specificity, train_sensitivity, train_auroc, train_ppv, train_npv = calculate_classification_metrics(y_class_train, y_train_pred_class, "Training Set ")

# Calculate and print regression metrics for training set
train_mse, train_r2 = calculate_regression_metrics(y_reg_train, y_train_pred_reg, "Training Set ")

# Save confusion matrices to CSV
save_to_csv(train_conf_matrix, 'train_conf_matrix', 'Training Set ')
save_to_csv(test_conf_matrix, 'test_conf_matrix', 'Test Set ')

# Save classification reports to CSV
save_to_csv(train_classification_rep, 'train_classification_report', 'Training Set ')
save_to_csv(test_classification_rep, 'test_classification_report', 'Test Set ')

# Save regression metrics to CSV
save_to_csv(np.array([train_mse, train_r2]), 'train_regression_metrics', 'Training Set ')
save_to_csv(np.array([test_mse, test_r2]), 'test_regression_metrics', 'Test Set ')
