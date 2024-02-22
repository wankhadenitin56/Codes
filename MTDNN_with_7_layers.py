import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

# Set path and create directories
path = "D:\\Project Work\\AFTER COMPRY MODEL REFINMENT\\MODELS RESULTS\\multitasking model/4"
os.makedirs(path, exist_ok=True)

# Read data
file_path = "D:\\Project Work\\AFTER COMPRY MODEL REFINMENT\\working_data_nitin_new.csv"
df = pd.read_csv(file_path)
df.fillna(0)

# Separate features and labels
X = df.iloc[:, :1923].values
y_classification = df.iloc[:, 1923].values
y_regression = df.iloc[:, 1924].values

# Train-test split
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_classification, y_regression, test_size=0.2, random_state=None)

# Standardize input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Multitask model
input_layer = Input(shape=(1923,))
hidden_layer1 = Dense(500, activation='relu', kernel_regularizer=regularizers.l2(0.001))(input_layer)
hidden_layer2 = BatchNormalization()(hidden_layer1)
hidden_layer3 = Dropout(0.2)(hidden_layer2)

hidden_layer4 = Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.001))(hidden_layer3)
hidden_layer5 = Dropout(0.2)(hidden_layer4)
output_layer_class = Dense(1, activation='sigmoid', name='class_output')(hidden_layer5)
output_layer_reg = Dense(1, activation='linear', name='reg_output')(hidden_layer5)

# Create the model
model = Model(inputs=input_layer, outputs=[output_layer_class, output_layer_reg])

# Compile the model
model.compile(optimizer=Adam(lr=0.001),
              loss={'class_output': 'binary_crossentropy', 'reg_output': 'mean_squared_error'},
              metrics={'class_output': 'accuracy'})

# Define early stopping
early_stopping = EarlyStopping(monitor='val_reg_output_loss', patience=10, restore_best_weights=True)

# Train the model
model.fit(X_train_scaled, {'class_output': y_class_train, 'reg_output': y_reg_train},
          validation_data=(X_test_scaled, {'class_output': y_class_test, 'reg_output': y_reg_test}),
          epochs=500, batch_size=32, callbacks=[early_stopping], verbose=1)

# Evaluate the model
predictions = model.predict(X_test_scaled)
class_pred = np.round(predictions[0]).flatten()
reg_pred = predictions[1].flatten()

# Print evaluation metrics
print("Classification Accuracy:", accuracy_score(y_class_test, class_pred))
print("Regression R2 Score:", r2_score(y_reg_test, reg_pred))
print("Regression MSE:", mean_squared_error(y_reg_test, reg_pred))
