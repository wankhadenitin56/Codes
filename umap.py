
print("%%%%%%%%%%%%%   20 feature models Boruta %%%%%%%% ")
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, roc_auc_score, confusion_matrix, classification_report, average_precision_score
from boruta import BorutaPy
import joblib
import seaborn as sns
 
# Set the path
# path = "D:\\Project Work\\MAIN\\Boruta_20_feature\\9" 
path = "D:\\Project Work\\AFTER COMPRY MODEL REFINMENT\\MODELS RESULTS\\extra\\5"
os.makedirs(path, exist_ok=True)

# Load the dataset
n = pd.read_csv("C:/Users/wankh/OneDrive/Desktop/project work/Working_on/Maindpp4.csv")
print(n.shape)

print("%%%%%%%%%%%%%%%%%%%%%  bar plot of the distribution of the target variable %%%%%%%%%%%%%%%%%%%%%%%%")

target_counts = n["target"].value_counts()

plt.figure(figsize=(8, 6))
sns.set(style="whitegrid")
sns.barplot(x=target_counts.index, y=target_counts.values)
plt.title("Distribution of Target Variable")
plt.xlabel("Target Value")
plt.ylabel("Count")
plt.tight_layout()

# Save the plot as an image
plot_path = os.path.join(path, 'target_distribution.png')
plt.savefig(plot_path)
plt.show()
# Data splitting
print("Data splitting...")

train_set, test_set = train_test_split(n, test_size=0.3)
# Extracting features and target labels for train  set
X_train = train_set.drop("target", axis=1)
y_train = train_set["target"]

# Extracting features and target labels for test set
X_test = test_set.drop("target", axis=1)
y_test = test_set["target"]

# Save X_train, y_train, X_test, y_test data
X_train_path = os.path.join(path, 'X_train.csv')
X_test_path = os.path.join(path, 'X_test.csv')
y_train_path = os.path.join(path, 'y_train.csv')
y_test_path = os.path.join(path, 'y_test.csv')

# Save X_train and X_test as CSV
X_train.to_csv(X_train_path, index=False)
X_test.to_csv(X_test_path, index=False)

# Save y_train and y_test as CSV
pd.DataFrame(y_train, columns=['target']).to_csv(y_train_path, index=False)
pd.DataFrame(y_test, columns=['target']).to_csv(y_test_path, index=False)

# Zero variance feature selection
print("Performing zero variance feature selection...")

zv = VarianceThreshold(threshold=0)
X_train_zv = zv.fit_transform(X_train)
print(X_train_zv.shape)
X_test_zv = zv.transform(X_test)

# Create a new folder for transformed matrices
transformed_folder = os.path.join(path, 'transformed_data')
os.makedirs(transformed_folder, exist_ok=True)

# Define file paths for transformed X_train_zv and X_test_zv matrices
X_train_zv_path = os.path.join(transformed_folder, 'X_train_zv.csv')
X_test_zv_path = os.path.join(transformed_folder, 'X_test_zv.csv')

# Convert the transformed matrices to DataFrames (optional)
X_train_zv_df = pd.DataFrame(X_train_zv, columns=X_train.columns[zv.get_support()])
X_test_zv_df = pd.DataFrame(X_test_zv, columns=X_train.columns[zv.get_support()])

# Save transformed matrices as CSV in the new folder
X_train_zv_df.to_csv(X_train_zv_path, index=False)
X_test_zv_df.to_csv(X_test_zv_path, index=False)

# Save the features after removing zero variance
selected_features_path = os.path.join(transformed_folder, 'selected_features.csv')
pd.DataFrame(X_train.columns[zv.get_support()], columns=['Selected Features']).to_csv(selected_features_path, index=False)

# Highly correlated features removal
print("Handling highly correlated features...")
cor_matrix_train = X_train.corr()

# Identify highly correlated features to be removed
cor_remove_train = [column for column in cor_matrix_train.columns if any(cor_matrix_train[column] > 0.85)]

# Keep only one feature from each group of highly correlated features
features_to_keep = set()
features_to_remove = []
for column in cor_remove_train:
    correlated_group = set([column] + [col for col in cor_matrix_train.index if cor_matrix_train.loc[col, column] > 0.85])
    feature_to_keep = correlated_group.pop()  
    features_to_keep.add(feature_to_keep)
    features_to_remove.extend(correlated_group)

# Remove highly correlated features from both training and test sets
X_train_final = X_train.drop(columns=features_to_remove)
print(X_train_final.shape)
X_test_final = X_test.drop(columns=features_to_remove)
print(X_test_final.shape)
# Save the features to keep and the removed features
remaining_features_path = os.path.join(transformed_folder, 'remaining_features.csv')
removed_features_path = os.path.join(transformed_folder, 'removed_features.csv')

pd.DataFrame(list(features_to_keep), columns=['Remaining Features']).to_csv(remaining_features_path, index=False)
pd.DataFrame(features_to_remove, columns=['Removed Features']).to_csv(removed_features_path, index=False)

# Normalize the data
print("Normalizing the data....")
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train_final)
X_test_normalized = scaler.transform(X_test_final)
# Perform UMAP dimension reduction
from umap import UMAP
import umap
import plotly.express as px  # Correct import statement

reducer = umap.UMAP() 

X_train_umap = reducer.fit_transform(X_train_normalized)
X_test_umap = reducer.transform(X_test_normalized)

# Plot UMAP for training data
fig_train = px.scatter(
    x=X_train_umap[:, 0], 
    y=X_train_umap[:, 1], 
    color=y_train,
    labels={'color': 'Target'},
    title='UMAP Plot for Training Data'
)
fig_train.show()

# Plot UMAP for test data
fig_test = px.scatter(
    x=X_test_umap[:, 0], 
    y=X_test_umap[:, 1], 
    color=y_test,
    labels={'color': 'Target'},
    title='UMAP Plot for Test Data'
)
fig_test.show()

# Dimension reduced datasets 
print(X_train_umap.shape)
print(X_test_umap.shape)