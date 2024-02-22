
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder,MinMaxScaler
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import (accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score, hamming_loss,
                             roc_curve, auc, precision_recall_curve)
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow import keras

# Set the path
path = ("D:/Project Work/MAIN NITIN PROJECT/Deep Learning/7")
os.makedirs(path, exist_ok=True)

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

sc = MinMaxScaler()
#sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Define the model architecture
keep_prob = 0.6
model = Sequential()
model.add(Dense(1923, input_dim=X_train.shape[1], activation='relu', kernel_regularizer='l2'))
#model.add(Dense(512, activation='relu'))
model.add(Dropout(keep_prob))
model.add(Dense(2, activation='sigmoid'))
opt = keras.optimizers.Adam(learning_rate=0.0002)  #0.001

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train the model
history = model.fit(X_train, Y_train, epochs=5000, batch_size=120, validation_data=(X_test, Y_test), verbose=2)

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
print("Accuracy test set:", accuracy_test)
print("Accuracy test set:", accuracy_train)

# Precision, Recall, Specificity
precision_train = precision_score(Y_train.argmax(axis=1), model.predict(X_train).argmax(axis=1), average=None)
recall_train = recall_score(Y_train.argmax(axis=1), model.predict(X_train).argmax(axis=1), average=None)
confusion_train = confusion_matrix(Y_train.argmax(axis=1), model.predict(X_train).argmax(axis=1))
ppv_train = precision_train
npv_train = np.diag(confusion_train) / (np.sum(confusion_train, axis=1) - np.diag(confusion_train))

precision_test = precision_score(Y_test_classes, Y_pred_classes, average=None)
recall_test = recall_score(Y_test_classes, Y_pred_classes, average=None)
confusion_test = confusion_matrix(Y_test_classes, Y_pred_classes)
ppv_test = precision_test
npv_test = np.diag(confusion_test) / (np.sum(confusion_test, axis=1) - np.diag(confusion_test))

# Calculate Specificity and Sensitivity for training and test sets
specificity_train = (confusion_train.diagonal() / (confusion_train.sum(axis=1) - confusion_train.diagonal())) * 100
sensitivity_train = recall_train * 100

specificity_test = (confusion_test.diagonal() / (confusion_test.sum(axis=1) - confusion_test.diagonal())) * 100
sensitivity_test = recall_test * 100

# Loss and Hamming Loss
loss_train = model.evaluate(X_train, Y_train)[0]
hamming_distance_train = hamming_loss(Y_train, model.predict(X_train) > 0.5)

loss_test = model.evaluate(X_test, Y_test)[0]
hamming_distance_test = hamming_loss(Y_test, Y_pred > 0.5)

# AUC-ROC
roc_auc_train = roc_auc_score(Y_train, model.predict(X_train), multi_class='ovr')
roc_auc_test = roc_auc_score(Y_test, Y_pred, multi_class='ovr')

# Create a DataFrame for the evaluation results
evaluation_results = pd.DataFrame({
    "Metric": ["Training Accuracy", "Test Accuracy", "Training Precision (PPV)", "Test Precision (PPV)",
               "Training Recall (Sensitivity)", "Test Recall (Sensitivity)",
               "Training Specificity", "Test Specificity",
               "Training NPV", "Test NPV", "Training Loss", "Test Loss",
               "Training Hamming Distance", "Test Hamming Distance",
               "Training AUC-ROC", "Test AUC-ROC"],
    "Value": [accuracy_train * 100, accuracy_test * 100, ppv_train.mean() * 100, ppv_test.mean() * 100,
              recall_train.mean() * 100, recall_test.mean() * 100,
              specificity_train.mean(), specificity_test.mean(),
              npv_train.mean() * 100, npv_test.mean() * 100,
              loss_train, loss_test, hamming_distance_train, hamming_distance_test,
              roc_auc_train, roc_auc_test]
})

# Save the evaluation results to a CSV file
evaluation_results.to_csv(os.path.join(path, 'evaluation_results.csv'), index=False)

# Save the model results
model_results = {
    "Test Accuracy": accuracy_test,
    "Test Loss": loss_test,
    "Test Hamming Distance": hamming_distance_test,
    "Test AUC-ROC": roc_auc_test,
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

# Print Specificity and Sensitivity for train and test sets
print(f"Training Specificity: {specificity_train.mean():.2f}%")
print(f"Test Specificity: {specificity_test.mean():.2f}%")
print(f"Training Sensitivity (Recall): {sensitivity_train.mean():.2f}%")
print(f"Test Sensitivity (Recall): {sensitivity_test.mean():.2f}%")

 # Save the model to a file
model.save(os.path.join(path, 'Deep_learniing.h5'))