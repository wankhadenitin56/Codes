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
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, auc

# path = "D:\\Project Work\\MAIN\\Boruta_20_feature\\9"
path = "D:\\Project Work\\AFTER COMPRY MODEL REFINMENT\\MODELS RESULTS\\multitasking model/4"
os.makedirs(path, exist_ok=True)

file_path = "D:\\Project Work\\AFTER COMPRY MODEL REFINMENT\\working_data_nitin_new.csv"
df = pd.read_csv(file_path)
df.fillna(0)

X = df.iloc[:, :1923].values
y_classification = df.iloc[:, 1923].values
y_regression = df.iloc[:, 1924].values

X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_classification, y_regression, test_size=0.2)

# Standardize input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Multitask model
input_layer = Input(shape=(X_train.shape[1],))
x = Dense(1923, activation='relu')(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Classification task
class_output = Dense(1, activation='sigmoid', name='class_output')(x)

# Regression task
reg_output = Dense(1, activation='linear', name='reg_output')(x)

# Create the model
model = Model(inputs=input_layer, outputs=[class_output, reg_output])

# Compile the model
model.compile(optimizer=Adam(lr=0.001), 
              loss={'class_output': 'binary_crossentropy', 'reg_output': 'mean_squared_error'},
              metrics={'class_output': 'accuracy'})

# Train the model
model.fit(X_train_scaled, {'class_output': y_class_train, 'reg_output': y_reg_train},
          validation_data=(X_test_scaled, {'class_output': y_class_test, 'reg_output': y_reg_test}),
          epochs=500, batch_size=64, verbose=1)

# Evaluate the model
predictions = model.predict(X_test_scaled)
class_pred = np.round(predictions[0]).flatten()
reg_pred = predictions[1].flatten()

# Add your desired evaluation metrics here
print("Classification Accuracy:", accuracy_score(y_class_test, class_pred))
print("Regression R2 Score:", r2_score(y_reg_test, reg_pred))
