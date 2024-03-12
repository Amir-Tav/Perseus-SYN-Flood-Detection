import pandas as pd
import numpy as np
from xgboost import XGBClassifier  # Import XGBoost Classifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt  # Import pyplot for plotting


# Define the evaluation function
def evaluate_model(model, x_test, y_test):
    # Making predictions
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]  # Get probabilities for the positive class

    # Creating a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, columns=["Predicted Negative", "Predicted Positive"], 
                         index=["Actual Negative", "Actual Positive"])

    # Visualization of the confusion matrix with color
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')  # Use the 'Blues' colormap for light to dark blue
    plt.title('XGB model, Confusion Matrix ')
    plt.colorbar()
    tick_marks = np.arange(len(cm_df.columns))
    plt.xticks(tick_marks, cm_df.columns, rotation=45)
    plt.yticks(tick_marks, cm_df.index)

    # Labeling the plot
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max()/2. else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # Computing evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")


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

# XGBoost Model
Perseus = XGBClassifier(
    n_estimators=100,
    learning_rate=0.001,
    max_depth=5,
    min_child_weight=1,  # This is different from GradientBoostingClassifier
    subsample=0.9,
    colsample_bytree=1,  # This specifies the fraction of features to use
    objective='binary:logistic',  # Objective function to use
    use_label_encoder=False  # To avoid a deprecation warning
)


Perseus.fit(x_train, y_train, eval_metric='logloss')  # Added eval_metric to avoid warnings

# Evaluate the model
evaluate_model(Perseus, x_test, y_test)

# Save the model and scaler
joblib.dump(scaler, 'scaler.save')
joblib.dump(Perseus, 'XGB-model')
