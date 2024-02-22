import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


# path = "D:\\Project Work\\MAIN\\Boruta_20_feature\\9"
path = "D:\\Project Work\\AFTER COMPRY MODEL REFINMENT\\MODELS RESULTS\\multitasking model/1"
os.makedirs(path, exist_ok=True)

# Load the CSV file containing the model training history
history_path = "C:\\Users\\wankh\\Downloads\\history.csv"
history_df = pd.read_csv(history_path)

# Plot Accuracy
plt.figure(figsize=(8, 6))
plt.plot(history_df['class_output_accuracy'], label='Training Accuracy')
plt.plot(history_df['val_class_output_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(path, 'accuracy_plot.png'))  # Save the plot to the specified path
plt.show()

# Plot Loss
plt.figure(figsize=(8, 6))
plt.plot(history_df['loss'], label='Training Loss')
plt.plot(history_df['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(path, 'loss_plot.png'))  # Save the plot to the specified path
plt.show()

