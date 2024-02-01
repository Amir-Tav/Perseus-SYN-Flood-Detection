import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import joblib

# Load the test data
testData = pd.read_csv("D:/University/Year 3/CS3072-CS3605 FYP/Perseus-FYP/Datasets/Balanced-SYN-V2.csv")
data = testData.dropna().copy()

data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Selecting the top features identified by Random Forest
selected_features = [
     'Source Port', 'Destination Port',
    'Protocol', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Flow Bytes/s', 'Flow Packets/s'
]

x = data[selected_features]
y = data['Label']

# Scaling the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Encoding the target variable Normal vs SYN Flood
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_encoded, test_size=0.2, random_state=42)

# Reshape the input to be suitable for Conv1D (samples, time steps, features)
x_train_reshaped = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test_reshaped = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# CNN model
Perseus = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(x_train_reshaped.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
Perseus.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model
Perseus.fit(x_train_reshaped, y_train, batch_size=32, epochs=10)

# Evaluate the model
y_pred = Perseus.predict(x_test_reshaped)
y_pred_binary = (y_pred > 0.5).astype(int)

# Generating the confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)
cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])

print("Confusion Matrix:")
print(cm_df)

# Calculate different metrics
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Saving the scaler
joblib.dump(scaler, 'scaler.save')
# Saving the model
Perseus.save('CNN-model')
