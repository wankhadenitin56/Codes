################ DEEP LEARNING ################################
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import (accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score, hamming_loss,roc_curve, auc, precision_recall_curve)
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow import keras

# Set the path
path = ("D:/Project Work/MAIN NITIN PROJECT/Deep Learning/34")
os.makedirs(path, exist_ok=True)

# Set hyperparameters
batch_size = 64
epochs = 500
learning_rate = 0.001
keep_prob = 0.6
# Save hyperparameters to a text file
hyperparameters_text = f"Batch Size: {batch_size}\nEpochs: {epochs}\nLearning Rate: {learning_rate}\nKeep Probability: {keep_prob}"
with open(os.path.join(path, 'hyperparameters.txt'), 'w') as hyperparameters_file:
    hyperparameters_file.write(hyperparameters_text)

# Load your data and preprocess it
data = pd.read_csv("D:\\Project Work\\Maindpp4.csv")
data = data.iloc[:, 1:]
data = data.sample(frac=1).reset_index(drop=True)

X = data.drop(['target'], axis=1)
Y = data["target"]

encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(Y)
dummy_Y = np_utils.to_categorical(encoded_Y)

X, dummy_Y = shuffle(X, dummy_Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_Y, test_size=0.2, random_state=None)

# Save X_train, X_test, Y_train, Y_test as CSV files
X_train.to_csv(os.path.join(path, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(path, 'X_test.csv'), index=False)
pd.DataFrame(Y_train).to_csv(os.path.join(path, 'Y_train.csv'), index=False)
pd.DataFrame(Y_test).to_csv(os.path.join(path, 'Y_test.csv'), index=False)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Define the model architecture

model = Sequential()
model.add(Dense(1923, input_dim=X_train.shape[1], activation='relu', kernel_regularizer='l2'))
model.add(Dense(800, activation='relu'))
model.add(Dropout(keep_prob))
model.add(Dense(2, activation='sigmoid'))
opt = keras.optimizers.Adam(learning_rate=learning_rate)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train the model
history = model.fit(X_train, Y_train, epochs=epochs,batch_size=batch_size ,validation_data=(X_test, Y_test), verbose=2)

# Model summary
model.summary()
# Plot training history - Accuracy
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training history - Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(path, 'model_loss.png'))
plt.show()

# Evaluate the model on the test set
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_test_classes = np.argmax(Y_test, axis=1)

# Accuracy
accuracy_train = accuracy_score(Y_train.argmax(axis=1), model.predict(X_train).argmax(axis=1))
accuracy_test = accuracy_score(Y_test_classes, Y_pred_classes)
print("Accuracy train set:", accuracy_train)
print("Accuracy test set:", accuracy_test)

from sklearn.metrics import confusion_matrix

# Calculate specificity and sensitivity for training and test sets
def calculate_specificity_sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    return specificity, sensitivity

# Calculate PPV and NPV
def calculate_ppv_npv(y_true, y_pred):
    ppv = precision_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    npv = confusion[0, 0] / (confusion[0, 0] + confusion[1, 0])
    return ppv, npv

# Calculate and save specificity, sensitivity, PPV, and NPV
specificity_train, sensitivity_train = calculate_specificity_sensitivity(Y_train.argmax(axis=1), model.predict(X_train).argmax(axis=1))
specificity_test, sensitivity_test = calculate_specificity_sensitivity(Y_test_classes, Y_pred_classes)

ppv_train, npv_train = calculate_ppv_npv(Y_train.argmax(axis=1), model.predict(X_train).argmax(axis=1))
ppv_test, npv_test = calculate_ppv_npv(Y_test_classes, Y_pred_classes)

# Create a DataFrame for the metrics
metrics = pd.DataFrame({
    "Metric": ["Accuracy train set","Accuracy test set","Training Specificity", "Test Specificity", "Training Sensitivity", "Test Sensitivity",
               "Training PPV", "Test PPV", "Training NPV", "Test NPV"],
    "Value": [accuracy_train,accuracy_test,specificity_train, specificity_test, sensitivity_train, sensitivity_test,
              ppv_train, ppv_test, npv_train, npv_test]
})

# Save the metrics to a CSV file
metrics.to_csv(os.path.join(path, 'metrics.csv'), index=False)

# Print the metrics
print("Training Specificity:", specificity_train)
print("Test Specificity:", specificity_test)
print("Training Sensitivity (Recall):", sensitivity_train)
print("Test Sensitivity (Recall):", sensitivity_test)
print("Training PPV:", ppv_train)
print("Test PPV:", ppv_test)
print("Training NPV:", npv_train)
print("Test NPV:", npv_test)

# Loss and Hamming Loss
loss_train = model.evaluate(X_train, Y_train)[0]
hamming_distance_train = hamming_loss(Y_train, model.predict(X_train) > 0.5)

loss_test = model.evaluate(X_test, Y_test)[0]
hamming_distance_test = hamming_loss(Y_test, Y_pred > 0.5)

# AUC-ROC
roc_auc_train = roc_auc_score(Y_train, model.predict(X_train), multi_class='ovr')
roc_auc_test = roc_auc_score(Y_test, Y_pred, multi_class='ovr')

# Save the model results
model_results = {
    "Test Accuracy": accuracy_test,
    "Train Accuracy":accuracy_train,
    "Test Loss": loss_test,
    "Train loss ": loss_train,
    "Test Hamming Distance": hamming_distance_test,
    "train Hamming Distance":hamming_distance_train,
    "Test AUC-ROC": roc_auc_test,
    "Train AUC-ROC":roc_auc_train,
    
}

# Save the model results to a CSV file
model_results_df = pd.DataFrame(model_results, index=[0])
model_results_df.to_csv(os.path.join(path, 'model_results.csv'), index=False)

# Plot ROC curve for both train and test sets on a single graph
def plot_roc_curve(fpr_train, tpr_train, roc_auc_train, fpr_test, tpr_test, roc_auc_test, title):
    plt.figure()
    plt.plot(fpr_train, tpr_train, color='darkorange', lw=2, label='Train ROC curve (area = %0.2f)' % roc_auc_train)
    plt.plot(fpr_test, tpr_test, color='green', lw=2, label='Test ROC curve (area = %0.2f)' % roc_auc_test)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(path, 'ROC_curve.png'))
    plt.show()

fpr_train, tpr_train, _ = roc_curve(Y_train[:, 1], model.predict(X_train)[:, 1])
roc_auc_train = auc(fpr_train, tpr_train)

fpr_test, tpr_test, _ = roc_curve(Y_test[:, 1], Y_pred[:, 1])
roc_auc_test = auc(fpr_test, tpr_test)

plot_roc_curve(fpr_train, tpr_train, roc_auc_train, fpr_test, tpr_test, roc_auc_test, 'ROC Curve (Train and Test)')

# Plot Precision-Recall curve
def plot_precision_recall_curve(precision_train, recall_train, precision_test, recall_test, title):
    plt.figure()
    plt.plot(recall_train, precision_train, color='darkorange', lw=2, label='Train Precision-Recall curve')
    plt.plot(recall_test, precision_test, color='green', lw=2, label='Test Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(path, 'precision_recall_curve.png'))
    plt.show()

precision_train, recall_train, _ = precision_recall_curve(Y_train[:, 1], model.predict(X_train)[:, 1])
precision_test, recall_test, _ = precision_recall_curve(Y_test[:, 1], Y_pred[:, 1])

plot_precision_recall_curve(precision_train, recall_train, precision_test, recall_test, 'Precision-Recall Curve')

 # Save the model to a file
model.save(os.path.join(path, 'Deep_learniing.h5'))

from sklearn.metrics import classification_report

# Classification report for training and test sets
classification_report_train = classification_report(Y_train[:, 1], model.predict(X_train)[:, 1] > 0.5, target_names=['Class 0', 'Class 1'])
classification_report_test = classification_report(Y_test[:, 1], Y_pred[:, 1] > 0.5, target_names=['Class 0', 'Class 1'])
# Save classification reports to text files
with open(os.path.join(path, 'classification_report_train.txt'), 'w') as report_file:
    report_file.write("Classification Report for Training Set:\n")
    report_file.write(classification_report_train)

with open(os.path.join(path, 'classification_report_test.txt'), 'w') as report_file:
    report_file.write("Classification Report for Test Set:\n")
    report_file.write(classification_report_test)
print("Classification Report for Training Set:\n", classification_report_train)
print("Classification Report for Test Set:\n", classification_report_test)

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Function to plot ROC curve
def plot_roc_curve(fpr, tpr, roc_auc, title):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(path, 'roc_curve.png'))
    plt.show()

# Calculate ROC curve for training and test sets
fpr_train, tpr_train, _ = roc_curve(Y_train[:, 1], model.predict(X_train)[:, 1])
roc_auc_train = auc(fpr_train, tpr_train)

fpr_test, tpr_test, _ = roc_curve(Y_test[:, 1], Y_pred[:, 1])
roc_auc_test = auc(fpr_test, tpr_test)

# Plot ROC curves
plot_roc_curve(fpr_train, tpr_train, roc_auc_train, 'ROC Curve (Training Set)')
plot_roc_curve(fpr_test, tpr_test, roc_auc_test, 'ROC Curve (Test Set)')

