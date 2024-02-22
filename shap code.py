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
path = "D:\\Project Work\\AFTER COMPRY MODEL REFINMENT\\MODELS RESULTS\\shap\\2"
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

# Create a random forest classifier
rf = RandomForestClassifier(n_estimators=20, n_jobs=-1)
# Create BorutaPy instance
print("%%%%%%%%% Perform Boruta Feature Selection... %%%%%%%%%")
boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2)
# Fit BorutaPy on the dataset
boruta_selector.fit(X_train_normalized, train_set["target"])

# Get the indices of the selected features by Boruta
selected_feature_indices_boruta = np.where(boruta_selector.support_)[0]

# Get the top 20 selected feature indices
top_20_feature_indices_boruta = selected_feature_indices_boruta[:20]
print(top_20_feature_indices_boruta)

# Get the selected feature names
selected_features_boruta = X_train.columns[top_20_feature_indices_boruta]
print(type(selected_feature_indices_boruta))
print(selected_feature_indices_boruta.shape)

# Get the rankings of selected features
selected_ranks_boruta = np.arange(1, len(selected_features_boruta) + 1)

# Plot the graph for top 20 features selected by Boruta
print("Plotting the graph for top 20 features selected by Boruta...")

plt.figure(figsize=(10, 6))
plt.barh(range(len(selected_features_boruta)), selected_ranks_boruta, align='center')
plt.yticks(range(len(selected_features_boruta)), selected_features_boruta)
plt.xlabel('Feature Ranking')
plt.ylabel('Selected Features')
plt.title('Top 20 Features Selected by Boruta')
plt.gca().invert_yaxis()
top_20_features_boruta_graph_path = os.path.join(path, 'top_20_features_boruta_graph.png')
plt.savefig(top_20_features_boruta_graph_path)
plt.show()
# Save the indices of the selected features by Boruta in a TXT file
selected_features_boruta_path = os.path.join(path, 'selected_features_boruta.txt')
with open(selected_features_boruta_path, 'w') as txt_file:
    for index in selected_feature_indices_boruta:
        txt_file.write(str(index) + '\n')

print("Selected feature indices saved to:", selected_features_boruta_path)
# Save the feature names to a text file
with open(os.path.join(path, 'top_20_feature_names_Boruta.txt'), 'w') as file:
    for feature_index in top_20_feature_indices_boruta:
        file.write(train_set.drop("target", axis=1).columns[feature_index] + '\n')
        
# Save the indices of the selected features by Boruta in a TXT file
selected_features_Boruta_path = os.path.join(path, 'selected_features_Boruta.txt')
with open(selected_features_Boruta_path, 'w') as txt_file:
    for index in selected_feature_indices_boruta:
        txt_file.write(str(index) + '\n')
        
X_train_selected = X_train_normalized[:,selected_feature_indices_boruta ]  
X_test_selected = X_test_normalized[:, selected_feature_indices_boruta]  
import shap
# SHAP feature importance
print("SHAP feature importance...")

# Train a new model on the selected features
model = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
model.fit(X_train_selected[:,top_20_feature_indices_boruta], y_train)  

# Create a SHAP explainer object using the trained model
explainer = shap.TreeExplainer(model)

# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_train_selected[:, top_20_feature_indices_boruta]) 
(np.shape(shap_values))
# Summary Plot with feature names
shap.summary_plot(shap_values, X_train_selected[:, top_20_feature_indices_boruta], feature_names=X_train.columns[top_20_feature_indices_boruta], plot_type="bar")  

# Detailed SHAP value plot with feature names
shap.summary_plot(shap_values[1], X_train_selected[:, top_20_feature_indices_boruta], feature_names=X_train.columns[top_20_feature_indices_boruta]) 

# Force Plot for a specific instance (e.g., first instance in the training set)
shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], X_train.iloc[0, top_20_feature_indices_boruta], feature_names=X_train.columns[top_20_feature_indices_boruta])

# Waterfall Plot for a specific instance (e.g., first instance in the training set)

shap.waterfall_plot(shap.Explanation(values=shap_values[1][0, :], base_values=explainer.expected_value[1], data=X_train.iloc[0, top_20_feature_indices_boruta], feature_names=X_train.columns[top_20_feature_indices_boruta]))

# Absolute Mean SHAP Values
abs_mean_shap_values = np.abs(shap_values).mean(axis=0)
print("Absolute Mean SHAP Values:")
print(abs_mean_shap_values)

# Get the selected feature names
selected_features_names = X_train.columns[top_20_feature_indices_boruta]

# Convert the absolute mean SHAP values to a DataFrame with feature names
abs_mean_shap_df = pd.DataFrame(data=abs_mean_shap_values, columns=selected_features_names)

# Save the DataFrame to a CSV file
abs_mean_shap_values_path = os.path.join(path, 'abs_mean_shap_values.csv')
abs_mean_shap_df.to_csv(abs_mean_shap_values_path, index=False)

# Choose a specific feature index for the dependence plot (e.g., the first feature)
feature_index_for_dependence_plot = top_20_feature_indices_boruta[0]
# Dependence Plot for a specific feature
shap.dependence_plot(ind=feature_index_for_dependence_plot, shap_values=shap_values[1], features=X_train_selected[:, top_20_feature_indices_boruta], feature_names=X_train.columns[top_20_feature_indices_boruta], show=False)
# Show the plot
plt.show()

