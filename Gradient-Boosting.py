import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import joblib

# Define the evaluation function
def evaluate_model(model, x_test, y_test):
    # Making predictions
    y_pred = model.predict(x_test)

    # Creating a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, columns=["Predicted Negative", "Predicted Positive"], 
                         index=["Actual Negative", "Actual Positive"])

    # Computing evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    print("Confusion Matrix:\n", cm_df)
    print("\nEvaluation Metrics:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

# Load the dataset
testData = pd.read_csv("D:/University/Year 3/CS3072-CS3605 FYP/Perseus-FYP/Datasets/Balanced-SYN-V2.csv")

# Data preprocessing
testData = testData.replace([np.inf, -np.inf], np.nan)
data = testData.dropna().copy()
data.columns = data.columns.str.strip()

selected_features = [
    'Source Port', 'Destination Port', 'Protocol', 'Flow Duration', 
    'Total Fwd Packets', 'Total Backward Packets', 'Flow Bytes/s', 
    'Flow Packets/s'

]

x = data[selected_features]
y = data['Label']

# Scaling the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


# Encoding the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_encoded, test_size=0.2, random_state=42)

# Gradient Boosting Model
Perseus = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05158833257363777,
    max_depth=10,
    min_samples_split=50,
    min_samples_leaf=5,
    subsample=0.9,
    max_features=None
)

Perseus.fit(x_train, y_train)

# Evaluate the model
evaluate_model(Perseus, x_test, y_test)

# Save the model and scaler
joblib.dump(scaler, 'scaler.save')
joblib.dump(Perseus, 'GB-model')




















#final model for deployment 
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# import joblib


# testData = pd.read_csv("D:/University/Year 3/CS3072-CS3605 FYP/Perseus-FYP/Datasets/Balanced_Syn.csv")

# # Identify and remove rows with infinity or NaN values
# testData = testData.replace([np.inf, -np.inf], np.nan)
# data = testData.dropna().copy()

# # Ensure there are no leading or trailing whitespaces in column names
# data.columns = data.columns.str.strip()

# # Selecting the features  #total of 8 features
# selected_features = [

#     'Source Port', 'Destination Port',
#     'Protocol', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
#     'Flow Bytes/s', 'Flow Packets/s'
    
#     ]

# x = data[selected_features]
# y = data['Label']

# # Scaling the features
# scaler = StandardScaler()
# x_scaled = scaler.fit_transform(x)

# # Encoding the target variable
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# # Splitting the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_encoded, test_size=0.2, random_state=42)

# Perseus = GradientBoostingClassifier(
#     n_estimators= 100,                      #number of trees 
#     learning_rate= 0.05158833257363777,                     #low values for robust learning 
#     max_depth= 5,                           #deep trees for more complex patterns 
#     min_samples_split= 50,                   #threshold for splitting nodes
#     min_samples_leaf= 5,                    #samples at each leaf
#     subsample= 0.9,                         #samples for fitting each tree for more robustness
#     max_features= None                      #find the best split
# )


# # Gradient Boosting Model
# Perseus.fit(x_train, y_train)

# # Making predictions
# y_pred = Perseus.predict(x_test)


# # Evaluating the model
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='binary')
# recall = recall_score(y_test, y_pred, average='binary')
# f1 = f1_score(y_test, y_pred, average='binary')

# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)

# joblib.dump(scaler, 'scaler.save')
# joblib.dump(Perseus, 'GB-model')
