

print("%%%%%%%%%%%%%   Whole ML model %%%%%%%% ")
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
path = r"D:\Project Work\AFTER COMPRY MODEL REFINMENT\MODELS RESULTS\CCOMBINE_FEATURES_WITH_IMPORTANCE\Data_with_6899\2"
os.makedirs(path, exist_ok=True)

# Load the dataset
n = pd.read_csv("D:\Project Work\AFTER COMPRY MODEL REFINMENT\multitasking model data and models\dpp4_6899_ML.csv")
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

train_set, test_set = train_test_split(n, test_size=0.2)
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
print(X_train_zv)

X_test_zv = zv.transform(X_test)
print(X_test_zv)

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
X_test_final = X_test.drop(columns=features_to_remove)

# Save the features to keep and the removed features
remaining_features_path = os.path.join(transformed_folder, 'remaining_features.csv')
removed_features_path = os.path.join(transformed_folder, 'removed_features.csv')

pd.DataFrame(list(features_to_keep), columns=['Remaining Features']).to_csv(remaining_features_path, index=False)
pd.DataFrame(features_to_remove, columns=['Removed Features']).to_csv(removed_features_path, index=False)


# Normalize the data
print("Normalizing the data...")
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train_final)
X_test_normalized = scaler.transform(X_test_final)

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
# Fit the RandomForestClassifier to your training data
rf.fit(X_train_normalized, y_train)
# Perform Recursive Feature Elimination (RFE)
print("%%%%%%%%% Perform Recursive Feature Elimination %%%%%%%%%")
num_top_features = 20
rfe_selector = RFE(rf, n_features_to_select=num_top_features, step=1)
rfe_selector.fit(X_train_normalized, y_train)

# Get the feature importances from the trained RandomForestClassifier
feature_importances = rf.feature_importances_

# Get the indices of the top 20 features based on importance scores
top_20_feature_indices_importance = np.argsort(feature_importances)[-num_top_features:][::-1]

# Create a DataFrame to store top 20 features, indices, and scores
top_20_features_df = pd.DataFrame(columns=['Feature', 'Index', 'Score'])

# Get the names of selected features
selected_features_importance = X_train.columns[top_20_feature_indices_importance]

# Get the importance scores of selected features
selected_importances = feature_importances[top_20_feature_indices_importance]

# Populate the DataFrame with top 20 features, indices, and scores
for i, (feature_name, index, score) in enumerate(zip(selected_features_importance, top_20_feature_indices_importance, selected_importances)):
    # Append the information to the DataFrame
    top_20_features_df = top_20_features_df.append({
        'Feature': feature_name,
        'Index': index,
        'Score': score
    }, ignore_index=True)

    print(f"{i + 1}. Feature: {feature_name}, Index: {index}, Score: {score:.4f}")

# Save the DataFrame to a CSV file or any other desired format
top_20_features_df.to_csv('top_20_features_importance.csv', index=False)
# Plot the graph for top features based on importance scores
print("Plot the graph for top features based on importance scores.......")

plt.figure(figsize=(10, 6))
plt.barh(range(num_top_features), selected_importances, align='center')
plt.yticks(range(num_top_features), selected_features_importance)
plt.xlabel('Feature Importance Score')
plt.ylabel('Selected Features')
plt.title('RFE Importance Feature  Scores')
plt.gca().invert_yaxis()

# Save the top features graph based on importance scores as an image
top_features_graph_importance_path = os.path.join(path, 'top_20_features_importance_graph.png')
plt.savefig(top_features_graph_importance_path)
plt.show()

total_selected_features_in_RFE = sum(rfe_selector.support_)

# Create a random forest classifier
rf = RandomForestClassifier(n_estimators=20, n_jobs=-1)

# Step 1: Use Boruta to select features
print("%%%%%%%%% Perform Boruta Feature Selection... %%%%%%%%%")
rf_for_boruta = RandomForestClassifier(n_estimators=20, n_jobs=-1)
boruta_selector = BorutaPy(rf_for_boruta, n_estimators='auto', verbose=2)

boruta_selector.fit(X_train_normalized, train_set["target"])

# Get the selected feature indices
selected_feature_indices_boruta = np.where(boruta_selector.support_)[0]

# Step 2: Apply Random Forest on selected features
rf_after_boruta = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf_after_boruta.fit(X_train_normalized[:, selected_feature_indices_boruta], train_set["target"])

# Step 3: Get feature importance scores from Random Forest
feature_importance_scores = rf_after_boruta.feature_importances_

# Step 4: Select the top 20 features based on importance scores
top_20_feature_indices_rf = np.argsort(feature_importance_scores)[::-1][:20]

# Print the top 20 feature names, indices, and scores
print("Top 20 Features:")
for i, index in enumerate(top_20_feature_indices_rf):
    feature_name = X_train.columns[selected_feature_indices_boruta[index]]
    score = feature_importance_scores[index]
    print(f"{i+1}. Feature: {feature_name}, Index: {selected_feature_indices_boruta[index]}, Score: {score:.4f}")

# Create a DataFrame to store top 20 features, indices, and scores
top_20_features_df = pd.DataFrame(columns=['Feature', 'Index', 'Score'])

# Print the top 20 feature names, indices, and scores
print("Top 20 Features:")
for i, index in enumerate(top_20_feature_indices_rf):
    feature_name = X_train.columns[selected_feature_indices_boruta[index]]
    score = feature_importance_scores[index]
    
    # Append the information to the DataFrame
    top_20_features_df = top_20_features_df.append({
        'Feature': feature_name,
        'Index': selected_feature_indices_boruta[index],
        'Score': score
    }, ignore_index=True)

    print(f"{i+1}. Feature: {feature_name}, Index: {selected_feature_indices_boruta[index]}, Score: {score:.4f}")

# Save the DataFrame to a CSV file or any other desired format
top_20_features_df.to_csv('top_20_boruta_features.csv', index=False)

# Plot the top 20 features
plt.figure(figsize=(10, 6))
plt.bar(range(len(top_20_feature_indices_rf)), feature_importance_scores[top_20_feature_indices_rf], align='center')
plt.xticks(range(len(top_20_feature_indices_rf)), [X_train.columns[selected_feature_indices_boruta[i]] for i in top_20_feature_indices_rf], rotation=32, ha='right')
plt.xlabel('Feature')
plt.ylabel('Importance Score')
plt.title('Top 20 Features Selected by after Boruta')

# Save the plot as an image
plot_path = os.path.join(path, 'top_20_features_plot.png')
plt.savefig(plot_path)
# Get the indices of the top 20 features selected by Boruta based on importance scores
selected_feature_indices_boruta = top_20_feature_indices_rf

# Get the indices of the top 20 features selected by RFE based on importance scores
selected_feature_indices_rfe = top_20_feature_indices_importance

# Combine the top 20 feature indices from RFE and Boruta, including the common ones
combined_feature_indices = np.unique(
    np.concatenate((selected_feature_indices_rfe, selected_feature_indices_boruta))
)

# Get the corresponding feature names for the combined top 20 feature indices
combined_feature_names = X_train.columns[combined_feature_indices]

# Print the combined top 20 feature names
print("Combined Top 20 Feature Names:")
for feature_name in combined_feature_names:
    print(feature_name)

# Create new training and test datasets with the combined selected features
X_train_combined = X_train_normalized[:, combined_feature_indices]
X_test_combined = X_test_normalized[:, combined_feature_indices]
print(X_train_combined.shape)
print(X_test_combined.shape)
# Define the paths to save the combined selected features
X_train_combined_path = os.path.join(path, 'X_train_combined.csv')
X_test_combined_path = os.path.join(path, 'X_test_combined.csv')

# Convert the combined selected features arrays to DataFrames
X_train_combined_df = pd.DataFrame(X_train_combined, columns=combined_feature_names)
X_test_combined_df = pd.DataFrame(X_test_combined, columns=combined_feature_names)

# Save the combined selected features DataFrames as CSV files
X_train_combined_df.to_csv(X_train_combined_path, index=False)
X_test_combined_df.to_csv(X_test_combined_path, index=False)

print("Combined selected features saved as CSV files:")
print("X_train_combined saved to:", X_train_combined_path)
print("X_test_combined saved to:", X_test_combined_path)

# Step 9: Build an SVM model using the common features
print("Build an SVM model using the common features......")
svm_param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.01, 0.1]
}

print("Performing SVM hyperparameter tuning...")
svm_grid_search = GridSearchCV(SVC(), svm_param_grid, cv=5, scoring='accuracy')
svm_grid_search.fit(X_train_combined, train_set["target"])
best_svm_params = svm_grid_search.best_params_

print("Best SVM Hyperparameters:", best_svm_params)

# Save the best SVM hyperparameters to a text file
svm_hyperparameters_path = os.path.join(path, 'best_svm_hyperparameters.txt')
with open(svm_hyperparameters_path, 'w') as f:
    f.write("Best SVM Hyperparameters:\n")
    for param, value in best_svm_params.items():
        f.write(f"{param}: {value}\n")
        
# Best SVM model from hyperparameter tuning
best_svm_model = svm_grid_search.best_estimator_

# Fit the best SVM model on the train data
best_svm_model.fit(X_train_combined, train_set["target"])

# Predict on the training set using the best SVM model
predictions_svm_train = best_svm_model.predict(X_train_combined)

# Predict on the test set using the best SVM model
predictions_svm_test = best_svm_model.predict(X_test_combined )

# Save predicted values for SVM model on training set
print("predicted values for SVM model on training set...")
svm_predictions_train_df = pd.DataFrame({
    "Predicted": predictions_svm_train,
    "Actual": train_set["target"]
})
svm_predictions_train_df.to_csv(os.path.join(path, 'svm_predictions_train.csv'), index=False)

# Save predicted values for SVM model on test set
print("predicted values for SVM model on test set....")
svm_predictions_test_df = pd.DataFrame({
    "Predicted": predictions_svm_test,
    "Actual": test_set["target"]
})
svm_predictions_test_df.to_csv(os.path.join(path, 'svm_predictions_test.csv'), index=False)

cm_svm_train = confusion_matrix(train_set["target"], predictions_svm_train)
cm_svm_test = confusion_matrix(test_set["target"], predictions_svm_test)

# Step 10: Evaluate the SVM model on the training and test sets
tn_svm_train, fp_svm_train, fn_svm_train, tp_svm_train = cm_svm_train.ravel()
specificity_train = tn_svm_train / (tn_svm_train + fp_svm_train)
sensitivity_train = tp_svm_train / (tp_svm_train + fn_svm_train)
svm_train_precision = precision_score(train_set["target"], predictions_svm_train)
svm_train_recall = recall_score(train_set["target"], predictions_svm_train)
svm_train_f1 = f1_score(train_set["target"], predictions_svm_train)
svm_train_accuracy = accuracy_score(train_set["target"], predictions_svm_train)

tn_svm_test, fp_svm_test, fn_svm_test, tp_svm_test = cm_svm_test.ravel()
specificity_test = tn_svm_test / (tn_svm_test + fp_svm_test)
sensitivity_test = tp_svm_test / (tp_svm_test + fn_svm_test)
svm_test_precision = precision_score(test_set["target"], predictions_svm_test)
svm_test_recall = recall_score(test_set["target"], predictions_svm_test)
svm_test_f1 = f1_score(test_set["target"], predictions_svm_test)
svm_test_accuracy = accuracy_score(test_set["target"], predictions_svm_test)

# Step 10.1: Calculate the probabilities for the positive class (class 1) for both training and test sets
svm_train_probs = best_svm_model.decision_function(X_train_combined )
svm_test_probs = best_svm_model.decision_function(X_test_combined )
# Step 10.2: Calculate the ROC AUC and Precision-Recall AUC for SVM model
print("ROC AUC and Precision-Recall AUC for SVM model....")
svm_train_roc_auc = roc_auc_score(train_set["target"], svm_train_probs)
svm_test_roc_auc = roc_auc_score(test_set["target"], svm_test_probs)

svm_train_pr_auc = average_precision_score(train_set["target"], svm_train_probs)
svm_test_pr_auc = average_precision_score(test_set["target"], svm_test_probs)
# Calculate Positive Predictive Value (PPV) for training and test sets
svm_ppv_train = tp_svm_train / (tp_svm_train + fp_svm_train)
svm_ppv_test = tp_svm_test / (tp_svm_test + fp_svm_test)

# Calculate Negative Predictive Value (NPV) for training and test sets
svm_npv_train = tn_svm_train / (tn_svm_train + fn_svm_train)
svm_npv_test = tn_svm_test / (tn_svm_test + fn_svm_test) 

# Print the evaluation results for SVM model on the training set
print("\nSVM Model Evaluation on the Training Set:")
print(f"Specificity (True Negative Rate): {specificity_train}")
print(f"Sensitivity (True Positive Rate): {sensitivity_train}")
print(f"Precision: {svm_train_precision}")
print(f"Recall: {svm_train_recall}")
print(f"F1 Score: {svm_train_f1}")
print(f"Accuracy: {svm_train_accuracy}")
print(f"ROC AUC: {svm_train_roc_auc}")
print(f"Precision-Recall AUC: {svm_train_pr_auc}")

# Print the evaluation results for SVM model on the test set
print("\nSVM Model Evaluation on the Test Set:")
print(f"Specificity (True Negative Rate): {specificity_test}")
print(f"Sensitivity (True Positive Rate): {sensitivity_test}")
print(f"Precision: {svm_test_precision}")
print(f"Recall: {svm_test_recall}")
print(f"F1 Score: {svm_test_f1}")
print(f"Accuracy: {svm_test_accuracy}")
print(f"ROC AUC: {svm_test_roc_auc}")
print(f"Precision-Recall AUC: {svm_test_pr_auc}")

# Save SVM model evaluation results on both training and test sets
print("SVM model evaluation results on both training and test sets......") 
svm_results = {
    "Train Specificity": specificity_train,
    "Train Sensitivity": sensitivity_train,
    "Train Precision": svm_train_precision,
    "Train Recall": svm_train_recall,
    "Train F1 Score": svm_train_f1,
    "Train Accuracy": svm_train_accuracy,
    "Train ROC AUC": svm_train_roc_auc,
    "Train PPV": svm_ppv_train,
    "Train NPV": svm_npv_train,
    "Train Precision-Recall AUC": svm_train_pr_auc,
    "Test Sensitivity": sensitivity_train,
    "Test Specificity": specificity_train,
    "Test Precision": svm_test_precision,
    "Test Recall": svm_test_recall,
    "Test F1 Score": svm_test_f1,
    "Test Accuracy": svm_test_accuracy,
    "Test ROC AUC": svm_test_roc_auc,
    "Test PPV": svm_ppv_test,
    "Test NPV": svm_npv_test,
   "Test Precision-Recall AUC": svm_test_pr_auc
}
# Save SVM model evaluation results
svm_results_path = os.path.join(path, 'svm_model_evaluation_results.txt')
with open(svm_results_path, 'w') as f:
    f.write("SVM Model Evaluation Results:\n")
    for metric, value in svm_results.items():
        f.write(f"{metric}: {value}\n")

# Save the best SVM model
svm_model_path = os.path.join(path, 'best_svm_model.pkl')
joblib.dump(best_svm_model, svm_model_path)

# Step 11: Build a Naive Bayes model using the combined features
print("Build a Naive Bayes model......")
nb_classifier = GaussianNB()

# Train the Naive Bayes classifier
nb_classifier.fit(X_train_combined, train_set["target"])

# Step 12: Hyperparameter Tuning
print("Performing Hyperparameter Tuning......")

# Define the range of values for 'var_smoothing'
params_NB = {'var_smoothing': np.logspace(-10, 0, num=50)}

# Create a Repeated Stratified K-Fold cross-validator
cv_method = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=999)

# Initialize GridSearchCV
gs_NB = GridSearchCV(estimator=nb_classifier, param_grid=params_NB, cv=cv_method, verbose=1, scoring='accuracy')

# Fit GridSearchCV
gs_NB.fit(X_train_combined, train_set["target"])

# Get the best parameter and score from GridSearchCV
best_var_smoothing = gs_NB.best_params_['var_smoothing']
best_score = gs_NB.best_score_
print("Best var_smoothing:", best_var_smoothing)
print("Best CV Score:", best_score)

# Save the best hyperparameter and its score to a text file
with open(os.path.join(path, 'nb_best_hyperparameter.txt'), 'w') as file:
    file.write("Best var_smoothing: {}\n".format(best_var_smoothing))
    file.write("Best CV Score: {}".format(best_score))
    # Predict on the training set
    predictions_nb_train = gs_NB.predict(X_train_combined)

    # Predict on the test set
    predictions_nb_test = gs_NB.predict(X_test_combined)


# Save predicted values for Naive Bayes model on training set
print("Save predicted values for Naive Bayes model.....")

nb_predictions_train_df = pd.DataFrame({
    "Predicted": predictions_nb_train,
    "Actual": train_set["target"]
})
nb_predictions_train_df.to_csv(os.path.join(path, 'nb_predictions_train.csv'), index=False)

# Save predicted values for Naive Bayes model on test set
nb_predictions_test_df = pd.DataFrame({
    "Predicted": predictions_nb_test,
    "Actual": test_set["target"]
})
nb_predictions_test_df.to_csv(os.path.join(path, 'nb_predictions_test.csv'), index=False)

# Step 12: Evaluate the Naive Bayes model on the training and test sets
nb_train_precision = precision_score(train_set["target"], predictions_nb_train)
nb_train_recall = recall_score(train_set["target"], predictions_nb_train)
nb_train_f1 = f1_score(train_set["target"], predictions_nb_train)
nb_train_accuracy = accuracy_score(train_set["target"], predictions_nb_train)

nb_test_precision = precision_score(test_set["target"], predictions_nb_test)
nb_test_recall = recall_score(test_set["target"], predictions_nb_test)
nb_test_f1 = f1_score(test_set["target"], predictions_nb_test)
nb_test_accuracy = accuracy_score(test_set["target"], predictions_nb_test)

# Calculate the probabilities for the positive class (class 1) for both training and test sets
nb_train_probs = nb_classifier.predict_proba(X_train_combined)[:, 1]
nb_test_probs = nb_classifier.predict_proba(X_test_combined)[:, 1]

# Calculate the ROC AUC and Precision-Recall AUC for Naive Bayes model
nb_train_roc_auc = roc_auc_score(train_set["target"], nb_train_probs)
nb_test_roc_auc = roc_auc_score(test_set["target"], nb_test_probs)

nb_train_pr_auc = average_precision_score(train_set["target"], nb_train_probs)
nb_test_pr_auc = average_precision_score(test_set["target"], nb_test_probs)

# confusion matrix for NB model
cm_nb_train = confusion_matrix(train_set["target"], predictions_nb_train)
cm_nb_test = confusion_matrix(test_set["target"], predictions_nb_test)

# Training Set
tn_train, fp_train, fn_train, tp_train = cm_nb_train.ravel()
specificity_train = tn_train / (tn_train + fp_train)
sensitivity_train = tp_train / (tp_train + fn_train)

# Test Set
tn_test, fp_test, fn_test, tp_test = cm_nb_test.ravel()
specificity_test = tn_test / (tn_test + fp_test)
sensitivity_test = tp_test / (tp_test + fn_test)
# Calculate Positive Predictive Value (PPV) for training and test sets
nb_ppv_train = tp_train / (tp_train + fp_train)
nb_ppv_test = tp_test / (tp_test + fp_test)

# Calculate Negative Predictive Value (NPV) for training and test sets
nb_npv_train = tn_train / (tn_train + fn_train)
nb_npv_test = tn_test / (tn_test + fn_test) 

# Print the evaluation results for Naive Bayes model on the training set
print("Print the evaluation results for Naive Bayes model......")

print("\nNaive Bayes Model Evaluation on the Training Set:")
print(f"Specificity (True Negative Rate): {specificity_train}")
print(f"Sensitivity (True Positive Rate): {sensitivity_train}")
print(f"Precision: {nb_train_precision}")
print(f"Recall: {nb_train_recall}")
print(f"F1 Score: {nb_train_f1}")
print(f"Accuracy: {nb_train_accuracy}")
print(f"ROC AUC: {nb_train_roc_auc}")
print(f"Precision-Recall AUC: {nb_train_pr_auc}")

# Print the evaluation results for Naive Bayes model on the test set
print("\nNaive Bayes Model Evaluation on the Test Set:")
print(f"Specificity (True Negative Rate): {specificity_test}")
print(f"Sensitivity (True Positive Rate): {sensitivity_test}")
print(f"Precision: {nb_test_precision}")
print(f"Recall: {nb_test_recall}")
print(f"F1 Score: {nb_test_f1}")
print(f"Accuracy: {nb_test_accuracy}")
print(f"ROC AUC: {nb_test_roc_auc}")
print(f"Precision-Recall AUC: {nb_test_pr_auc}")

# Save Naive Bayes model evaluation results on both training and test sets
nb_results = {
    "Train Specificity": specificity_train,
    "Train Sensitivity": sensitivity_train,
    "Train Precision": nb_train_precision,
    "Train Recall": nb_train_recall,
    "Train F1 Score": nb_train_f1,
    "Train Accuracy": nb_train_accuracy,
    "Train ROC AUC": nb_train_roc_auc,
    "Train PPV": nb_ppv_train,
    "Train NPV": nb_npv_train,
    "Train Precision-Recall AUC": nb_train_pr_auc,
    "Test Specificity": specificity_train,
    "Test Sensitivity": sensitivity_train,
    "Test Precision": nb_test_precision,
    "Test Recall": nb_test_recall,
    "Test F1 Score": nb_test_f1,
    "Test Accuracy": nb_test_accuracy,
    "Test ROC AUC": nb_test_roc_auc,
    "Test PPV": nb_ppv_test,
    "Test NPV": nb_npv_test,
    "Test Precision-Recall AUC": nb_test_pr_auc
}
# Save NB model evaluation results
nb_results_path = os.path.join(path, 'NB_evaluation_results.txt')
with open(nb_results_path, 'w') as f:
    f.write("NB Evaluation Results:\n")
    for metric, value in nb_results.items():
        f.write(f"{metric}: {value}\n")
        
# Save the Naive Bayes model
nb_model_path = os.path.join(path, 'best_naive_bayes_model.pkl')
joblib.dump(nb_classifier, nb_model_path)

# Step 13: Build a Random Forest model using the common features

print("Build a Random Forest model....")
rf_param_grid = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': [20, 30, 40],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [2, 4, 6]
}
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='accuracy')
rf_grid_search.fit(X_train_combined, train_set["target"])
best_rf_params = rf_grid_search.best_params_

print("Best rf Hyperparameters:", best_rf_params)

# Save the best rf hyperparameters to a text file
rf_hyperparameters_path = os.path.join(path, 'best_rf_hyperparameters.txt')
with open(rf_hyperparameters_path, 'w') as f:
    f.write("Best rf Hyperparameters:\n")
    for param, value in best_rf_params.items():
        f.write(f"{param}: {value}\n")
      
# Best Random Forest model from hyperparameter tuning
best_rf_model = rf_grid_search.best_estimator_

# Fit the best Random Forest model on the train data
best_rf_model.fit(X_train_combined, train_set["target"])

# Predict on the training set using the best Random Forest model
predictions_rf_train = best_rf_model.predict(X_train_combined)

# Predict on the test set using the best Random Forest model
predictions_rf_test = best_rf_model.predict(X_test_combined)

# Save predicted values for Random Forest model on training set
rf_predictions_train_df = pd.DataFrame({
    "Predicted": predictions_rf_train,
    "Actual": train_set["target"]
})
rf_predictions_train_df.to_csv(os.path.join(path, 'rf_predictions_train.csv'), index=False)

# Save predicted values for Random Forest model on test set
rf_predictions_test_df = pd.DataFrame({
    "Predicted": predictions_rf_test,
    "Actual": test_set["target"]
})
rf_predictions_test_df.to_csv(os.path.join(path, 'rf_predictions_test.csv'), index=False)

cm_rf_train = confusion_matrix(train_set["target"], predictions_rf_train)
cm_rf_test = confusion_matrix(test_set["target"], predictions_rf_test)
tn_rf_train, fp_rf_train, fn_rf_train, tp_rf_train = cm_rf_train.ravel()
specificity_rf_train = tn_rf_train / (tn_rf_train + fp_rf_train)
sensitivity_rf_train = tp_rf_train / (tp_rf_train + fn_rf_train)

tn_rf_test, fp_rf_test, fn_rf_test, tp_rf_test = cm_rf_train.ravel()
specificity_rf_test = tn_rf_test / (tn_rf_test + fp_rf_test)
sensitivity_rf_test = tp_rf_test / (tp_rf_test + fn_rf_test)


# Step 14: Evaluate the Random Forest model on the training and test sets
print("Evaluate the Random Forest model on the training and test sets.....")

rf_train_precision = precision_score(train_set["target"], predictions_rf_train)
rf_train_recall = recall_score(train_set["target"], predictions_rf_train)
rf_train_f1 = f1_score(train_set["target"], predictions_rf_train)
rf_train_accuracy = accuracy_score(train_set["target"], predictions_rf_train)

rf_test_precision = precision_score(test_set["target"], predictions_rf_test)
rf_test_recall = recall_score(test_set["target"], predictions_rf_test)
rf_test_f1 = f1_score(test_set["target"], predictions_rf_test)
rf_test_accuracy = accuracy_score(test_set["target"], predictions_rf_test)

# Step 14.1: Calculate the probabilities for the positive class (class 1) for both training and test sets
rf_train_probs = best_rf_model.predict_proba(X_train_combined)[:, 1]
rf_test_probs = best_rf_model.predict_proba(X_test_combined)[:, 1]

# Step 14.2: Calculate the ROC AUC and Precision-Recall AUC for Random Forest model
rf_train_roc_auc = roc_auc_score(train_set["target"], rf_train_probs)
rf_test_roc_auc = roc_auc_score(test_set["target"], rf_test_probs)

rf_train_pr_auc = average_precision_score(train_set["target"], rf_train_probs)
rf_test_pr_auc = average_precision_score(test_set["target"], rf_test_probs)
# Calculate Negative Predictive Value (NPV) for training and test sets
npv_rf_train = tn_rf_train / (tn_rf_train + fn_rf_train)
npv_rf_test = tn_rf_test / (tn_rf_test + fn_rf_test)

# Calculate Positive Predictive Value (PPV) for training and test sets
ppv_rf_train = tp_rf_train / (tp_rf_train + fp_rf_train)
ppv_rf_test = tp_rf_test / (tp_rf_test + fp_rf_test)

# Print the evaluation results for Random Forest model on the training set, including NPV and PPV
print("\nRandom Forest Model Evaluation on the Training Set:")
print(f"Specificity: {specificity_rf_train}")
print(f"Sensitivity: {sensitivity_rf_train}")
print(f"PPV: {ppv_rf_train}")
print(f"NPV: {npv_rf_train}")
print(f"Precision: {rf_train_precision}")
print(f"Recall: {rf_train_recall}")
print(f"F1 Score: {rf_train_f1}")
print(f"Accuracy: {rf_train_accuracy}")
print(f"ROC AUC: {rf_train_roc_auc}")
print(f"Precision-Recall AUC: {rf_train_pr_auc}")

# Print the evaluation results for Random Forest model on the test set, including NPV and PPV
print("\nRandom Forest Model Evaluation on the Test Set:")
print(f"Specificity: {specificity_rf_test}")
print(f"Sensitivity: {sensitivity_rf_test}")
print(f"PPV: {ppv_rf_test}")
print(f"NPV: {npv_rf_test}")
print(f"Precision: {rf_test_precision}")
print(f"Recall: {rf_test_recall}")
print(f"F1 Score: {rf_test_f1}")
print(f"Accuracy: {rf_test_accuracy}")
print(f"ROC AUC: {rf_test_roc_auc}")
print(f"Precision-Recall AUC: {rf_test_pr_auc}")

# Save Random Forest model evaluation results on both training and test sets, including NPV and PPV
print("Save Random Forest model evaluation results on both training and test sets, including NPV and PPV.....")
rf_results = {
    "Train Sensitivity": specificity_rf_train,
    "Train Specificity": sensitivity_rf_train,
    "Train PPV": ppv_rf_train,
    "Train NPV": npv_rf_train,
    "Train Precision": rf_train_precision,
    "Train Recall": rf_train_recall,
    "Train F1 Score": rf_train_f1,
    "Train Accuracy": rf_train_accuracy,
    "Train ROC AUC": rf_train_roc_auc,
    "Train Precision-Recall AUC": rf_train_pr_auc,
    "Test Specificity": specificity_rf_test,
    "Test Sensitivity": sensitivity_rf_test,
    "Test PPV": ppv_rf_test,
    "Test NPV": npv_rf_test,
    "Test Precision": rf_test_precision,
    "Test Recall": rf_test_recall,
    "Test F1 Score": rf_test_f1,
    "Test Accuracy": rf_test_accuracy,
    "Test ROC AUC": rf_test_roc_auc,
    "Test Precision-Recall AUC": rf_test_pr_auc
}      
# Save RF model evaluation results
RF_results_path = os.path.join(path, 'RF_evaluation_results.txt')
with open(RF_results_path, 'w') as f:
    f.write("RF Evaluation Results:\n")
    for metric, value in rf_results.items():
        f.write(f"{metric}: {value}\n")
        
# Save the best Random Forest model
rf_model_path = os.path.join(path, 'best_random_forest_model.pkl')
joblib.dump(best_rf_model, rf_model_path)

# Classification report function
def generate_classification_report(model, model_name, X, y):
    predictions = model.predict(X)
    class_report = classification_report(y, predictions)
    print(f"\nClassification Report for {model_name}:")
    print(class_report)
    # Save classification report to a text file
    report_path = os.path.join(path, f'{model_name.lower()}_classification_report.txt')
    with open(report_path, 'w') as file:
        file.write(f"Classification Report for {model_name}:\n")
        file.write(class_report)

# Generate classification reports for all models
print("Generate classification reports for all models.....")
generate_classification_report(best_svm_model, "SVM", X_test_combined, test_set["target"])
generate_classification_report(nb_classifier, "Naive_Bayes", X_test_combined, test_set["target"])
generate_classification_report(best_rf_model, "Random_Forest", X_test_combined, test_set["target"])

# Save predicted values and actual labels for SVM model
svm_predictions_df = pd.DataFrame({
    "Predicted": predictions_svm_test,
    "Actual": test_set["target"]
})
svm_predictions_df.to_csv(os.path.join(path, 'svm_predictions.csv'), index=False)

# Save predicted values and actual labels for Naive Bayes model
nb_predictions_df = pd.DataFrame({
    "Predicted": predictions_nb_test,
    "Actual": test_set["target"]
})
nb_predictions_df.to_csv(os.path.join(path, 'nb_predictions.csv'), index=False)

# Save predicted values and actual labels for Random Forest model
rf_predictions_df = pd.DataFrame({
    "Predicted": predictions_rf_test,
    "Actual": test_set["target"]
})
rf_predictions_df.to_csv(os.path.join(path, 'rf_predictions.csv'), index=False)

# Plot and save the ROC curve for SVM model
print("Plot and save the ROC curve for SVM model.....")

svm_decision_scores_train = best_svm_model.decision_function(X_train_combined)
svm_decision_scores_test = best_svm_model.decision_function(X_test_combined)
fpr_svm_train, tpr_svm_train, _ = roc_curve(train_set["target"], svm_decision_scores_train)
fpr_svm_test, tpr_svm_test, _ = roc_curve(test_set["target"], svm_decision_scores_test)
roc_auc_svm_train = roc_auc_score(train_set["target"], svm_decision_scores_train)
roc_auc_svm_test = roc_auc_score(test_set["target"], svm_decision_scores_test)

plt.figure(figsize=(8, 6))
plt.plot(fpr_svm_train, tpr_svm_train, color='darkorange', lw=2, label=f'Train ROC curve (area = {roc_auc_svm_train:.2f})')
plt.plot(fpr_svm_test, tpr_svm_test, color='green', lw=2, label=f'Test ROC curve  (area = {roc_auc_svm_test:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SVM Model')
plt.legend(loc='lower right')
plt.savefig(os.path.join(path, 'roc_curve_svm.png'))
plt.show()

# Plot and save the ROC curve for Naive Bayes model
print("Plot and save the ROC curve for Naive Bayes model....")
nb_probs_train = nb_classifier.predict_proba(X_train_combined)[:, 1]
nb_probs_test = nb_classifier.predict_proba(X_test_combined)[:, 1]
fpr_nb_train, tpr_nb_train, _ = roc_curve(train_set["target"], nb_probs_train)
fpr_nb_test, tpr_nb_test, _ = roc_curve(test_set["target"], nb_probs_test)
roc_auc_nb_train = roc_auc_score(train_set["target"], nb_probs_train)
roc_auc_nb_test = roc_auc_score(test_set["target"], nb_probs_test)

plt.figure(figsize=(8, 6))
plt.plot(fpr_nb_train, tpr_nb_train, color='darkorange', lw=2, label=f'Train ROC curve (area = {roc_auc_nb_train:.2f})')
plt.plot(fpr_nb_test, tpr_nb_test, color='green', lw=2, label=f'Test ROC curve (area = {roc_auc_nb_test:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Naive Bayes Model')
plt.legend(loc='lower right')
plt.savefig(os.path.join(path, 'roc_curve_nb.png'))
plt.show()

# Plot and save the ROC curve for Random Forest model
print("Plot and save the ROC curve for Random Forest model.....")
rf_probs_train = best_rf_model.predict_proba(X_train_combined)[:, 1]
rf_probs_test = best_rf_model.predict_proba(X_test_combined)[:, 1]
fpr_rf_train, tpr_rf_train, _ = roc_curve(train_set["target"], rf_probs_train)
fpr_rf_test, tpr_rf_test, _ = roc_curve(test_set["target"], rf_probs_test)
roc_auc_rf_train = roc_auc_score(train_set["target"], rf_probs_train)
roc_auc_rf_test = roc_auc_score(test_set["target"], rf_probs_test)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf_train, tpr_rf_train, color='darkorange', lw=2, label=f'Train ROC curve (area = {roc_auc_rf_train:.2f})')
plt.plot(fpr_rf_test, tpr_rf_test, color='green', lw=2, label=f'Test ROC curve (area = {roc_auc_rf_test:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest Model')
plt.legend(loc='lower right')
plt.savefig(os.path.join(path, 'roc_curve_rf.png'))
plt.show()

# Plot and save the confusion matrix for SVM model
print("Plot and save the confusion matrix for SVM model...")
cm_svm_train = confusion_matrix(train_set["target"], predictions_svm_train)
cm_svm_test = confusion_matrix(test_set["target"], predictions_svm_test)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.heatmap(cm_svm_train, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (Train) - SVM')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.subplot(1, 2, 2)
sns.heatmap(cm_svm_test, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (Test) - SVM')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.savefig(os.path.join(path, 'svm_confusion_matrix.png'))
plt.show()

# Plot and save the confusion matrix for Naive Bayes model
print("Plot and save the confusion matrix for Naive Bayes model....")
cm_nb_train = confusion_matrix(train_set["target"], predictions_nb_train)
cm_nb_test = confusion_matrix(test_set["target"], predictions_nb_test)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.heatmap(cm_nb_train, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (Train) - Naive Bayes')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.subplot(1, 2, 2)
sns.heatmap(cm_nb_test, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (Test) - Naive Bayes')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.savefig(os.path.join(path, 'naive_bayes_confusion_matrix.png'))
plt.show()

# Plot and save the confusion matrix for Random Forest model
print("Plot and save the confusion matrix for Random Forest model.......")
cm_rf_train = confusion_matrix(train_set["target"], predictions_rf_train)
cm_rf_test = confusion_matrix(test_set["target"], predictions_rf_test)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.heatmap(cm_rf_train, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (Train) - Random Forest')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.subplot(1, 2, 2)
sns.heatmap(cm_rf_test, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (Test) - Random Forest')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.savefig(os.path.join(path, 'random_forest_confusion_matrix.png'))
plt.show()


# Save results and metrics to a CSV file
print("Save results and metrics to a CSV file......")
results_df = pd.DataFrame({
    "Model": ["SVM", "Naive_Bayes", "Random_Forest"],
    "Train Precision": [svm_train_precision, nb_train_precision, rf_train_precision],
    "Train Recall": [svm_train_recall, nb_train_recall, rf_train_recall],
    "Train F1 Score": [svm_train_f1, nb_train_f1, rf_train_f1],
    "Train Accuracy": [svm_train_accuracy, nb_train_accuracy, rf_train_accuracy],
    "Train ROC AUC": [svm_train_roc_auc, nb_train_roc_auc, rf_train_roc_auc],
    "Train Precision-Recall AUC": [svm_train_pr_auc, nb_train_pr_auc, rf_train_pr_auc],
    "Train PPV":[svm_ppv_train,nb_ppv_train,ppv_rf_train],
    "Train NPV":[svm_npv_train,nb_npv_train,nb_ppv_train],
    "Test Precision": [svm_test_precision, nb_test_precision, rf_test_precision],
    "Test Recall": [svm_test_recall, nb_test_recall, rf_test_recall],
    "Test F1 Score": [svm_test_f1, nb_test_f1, rf_test_f1],
    "Test Accuracy": [svm_test_accuracy, nb_test_accuracy, rf_test_accuracy],
    "Test ROC AUC": [svm_test_roc_auc, nb_test_roc_auc, rf_test_roc_auc],
    "Test Precision-Recall AUC": [svm_test_pr_auc, nb_test_pr_auc, rf_test_pr_auc],
    "Test PPV":[svm_ppv_test,nb_ppv_test,ppv_rf_test],
    "Test NPV":[svm_npv_test,nb_npv_test,nb_ppv_test]
})
results_csv_path = os.path.join(path, 'model_results.csv')
results_df.to_csv(results_csv_path, index=False)


