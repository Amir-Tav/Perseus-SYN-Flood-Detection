# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
# import joblib

# # Load the test data
# testData = pd.read_csv("D:/University/Year 3/CS3072-CS3605 FYP/Perseus-FYP/Datasets/Balanced-SYN-V2.csv")
# data = testData.dropna().copy()

# data.replace([np.inf, -np.inf], np.nan, inplace=True)

# # Selecting the top features identified by Random Forest
# selected_features = [
#      'Source Port', 'Destination Port',
#     'Protocol', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
#     'Flow Bytes/s', 'Flow Packets/s'
# ]

# x = data[selected_features]
# y = data['Label']

# # Scaling the features
# scaler = StandardScaler()
# x_scaled = scaler.fit_transform(x)

# # Encoding the target variable Normal vs SYN Flood
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# # Splitting the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_encoded, test_size=0.2, random_state=42)

# # Neural network model
# Perseus = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(256, activation='relu', input_shape=(8,)), 
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(128, activation='linear'),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(64, activation='linear'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# # Compile the model
# Perseus.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# # Assuming x_train, y_train, x_test, y_test are properly prepared
# Perseus.fit(x_train, y_train, batch_size= 32, epochs=10)

# # Evaluate the model
# y_pred = Perseus.predict(x_test)
# y_pred_binary = (y_pred > 0.5).astype(int)

# # Generating the confusion matrix
# cm = confusion_matrix(y_test, y_pred_binary)
# cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])

# print("Confusion Matrix:")
# print(cm_df)


# # Calculate different metrics
# accuracy = accuracy_score(y_test, y_pred_binary)
# precision = precision_score(y_test, y_pred_binary)
# recall = recall_score(y_test, y_pred_binary)
# f1 = f1_score(y_test, y_pred_binary)

# print(f"Test Accuracy: {accuracy * 100:.2f}%")
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1 Score: {f1:.2f}")

# #saving the scaler
# joblib.dump(scaler, 'scaler.save')
# # saving the model 
# Perseus.save('NN-model')

######################################################pretty decent model ############################

# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import RobustScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
# import joblib

# # Load the test data
# testData = pd.read_csv("D:/University/Year 3/CS3072-CS3605 FYP/Perseus-FYP/Datasets/Balanced-SYN-V2.csv")
# data = testData.copy()

# # Handle missing and infinite values
# data.replace([np.inf, -np.inf], np.nan, inplace=True)
# data.dropna(subset=['Label'], inplace=True)  # Ensure labels are not NaN
# data.fillna(0, inplace=True)  # Fill NaN values with a placeholder (e.g., 0)

# # Selecting the top features identified by Random Forest
# selected_features = [
#     'Source Port', 'Destination Port', 'Protocol', 'Flow Duration', 
#     'Total Fwd Packets', 'Total Backward Packets', 'Flow Bytes/s', 'Flow Packets/s'
# ]

# x = data[selected_features]
# y = data['Label']

# # Scaling the features with RobustScaler
# scaler = RobustScaler()
# x_scaled = scaler.fit_transform(x)

# # Encoding the target variable Normal vs SYN Flood
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# # Splitting the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_encoded, test_size=0.2, random_state=42)

# # Neural network model
# Perseus = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(256, activation='relu', input_shape=(8,)), 
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# # Compile the model with updated learning rate parameter
# Perseus.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model
# Perseus.fit(x_train, y_train, batch_size=50, epochs=15)

# # Evaluate the model
# y_pred = Perseus.predict(x_test)
# y_pred_binary = (y_pred > 0.5).astype(int)

# # Generating the confusion matrix
# cm = confusion_matrix(y_test, y_pred_binary)
# cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])

# print("Confusion Matrix:")
# print(cm_df)

# # Calculate different metrics
# accuracy = accuracy_score(y_test, y_pred_binary)
# precision = precision_score(y_test, y_pred_binary)
# recall = recall_score(y_test, y_pred_binary)
# f1 = f1_score(y_test, y_pred_binary)

# print(f"Test Accuracy: {accuracy * 100:.2f}%")
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1 Score: {f1:.2f}")

# # Saving the scaler
# joblib.dump(scaler, 'scaler.save')

# # Saving the model
# Perseus.save('NN-model')

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
import matplotlib.pyplot as plt



# Load the test data
data = pd.read_csv("/mnt/data/Balanced-SYN-V2.csv")

# Handle missing and infinite values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(subset=['Label'], inplace=True)  # Ensure labels are not NaN
data.fillna(0, inplace=True)  # Fill NaN values with a placeholder (e.g., 0)

# Selecting the features
selected_features = [
    'Source Port', 'Destination Port', 'Protocol', 'Flow Duration', 
    'Total Fwd Packets', 'Total Backward Packets', 'Flow Bytes/s', 'Flow Packets/s'
]

x = data[selected_features]
y = data['Label']

# Scaling the features with RobustScaler
scaler = RobustScaler()
x_scaled = scaler.fit_transform(x)

# Encoding the target variable Normal vs SYN Flood
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_encoded, test_size=0.2, random_state=42)

# Neural network model
Perseus = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(8,)), 
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
Perseus.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
Perseus.fit(x_train, y_train, batch_size=50, epochs=15, verbose=0)  # Set verbose to 0 to suppress output

# Predict probabilities
y_pred_probs = Perseus.predict(x_test)

# Calculate ROC Curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_probs)
roc_auc = auc(fpr, tpr)

# Calculate Precision-Recall Curve
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_probs)
f1_scores = 2 * recall * precision / (recall + precision)
optimal_idx_pr = np.argmax(f1_scores)
optimal_threshold_pr = thresholds_pr[optimal_idx_pr]

# Find the optimal threshold from the ROC curve
optimal_idx_roc = np.argmax(tpr - fpr)
optimal_threshold_roc = thresholds_roc[optimal_idx_roc]

roc_auc, optimal_threshold_roc, optimal_threshold_pr


