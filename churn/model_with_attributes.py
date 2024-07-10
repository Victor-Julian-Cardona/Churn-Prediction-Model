import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
import numpy as np

# Load the data
data = pd.read_csv('Telco-Customer-Churn.csv')

# Preprocess the data
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

# Binary encoding
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 
               'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
               'PaperlessBilling', 'Churn']
for col in binary_cols:
    data[col] = data[col].apply(lambda x: 1 if x == 'Yes' else 0)

# One-hot encoding
onehot_cols = ['InternetService', 'Contract', 'PaymentMethod']
data = pd.get_dummies(data, columns=onehot_cols)

# Remove customerID
data = data.drop(columns=['customerID'])

# Define features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Separate numeric and categorical columns
num_cols = X.select_dtypes(include=['float64', 'int64']).columns
cat_cols = X.select_dtypes(include=['uint8']).columns

# Scale numeric data
scaler = StandardScaler()
X_train_resampled[num_cols] = scaler.fit_transform(X_train_resampled[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Fit and transform categorical data
encoder = OneHotEncoder(handle_unknown='ignore')
X_train_cat = encoder.fit_transform(X_train_resampled[cat_cols])
X_test_cat = encoder.transform(X_test[cat_cols])

# Concatenate numeric and categorical data
X_train_final = np.hstack((X_train_resampled[num_cols], X_train_cat.toarray()))
X_test_final = np.hstack((X_test[num_cols], X_test_cat.toarray()))

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_final, y_train_resampled)

# Predict with adjusted threshold
y_pred_proba = model.predict_proba(X_test_final)[:, 1]
threshold = 0.4  # Adjusted threshold
y_pred_adjusted = (y_pred_proba >= threshold).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_adjusted)
precision = precision_score(y_test, y_pred_adjusted)
recall = recall_score(y_test, y_pred_adjusted)
conf_matrix = confusion_matrix(y_test, y_pred_adjusted)
class_report = classification_report(y_test, y_pred_adjusted)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Extract and display feature importance
feature_names_num = num_cols.tolist()
feature_names_cat = encoder.get_feature_names_out(cat_cols).tolist()
feature_names = feature_names_num + feature_names_cat

if hasattr(model, 'coef_'):
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': model.coef_[0]})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    print(feature_importance)
else:
    print("Feature importance is not available for this model.")
