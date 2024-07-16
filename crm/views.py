import pandas as pd
from django.shortcuts import render, redirect
from joblib import load, dump
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import numpy as np

# Load the model and preprocessors
pipeline = load('churn/logistic_regression_model.pkl')
scaler = load('churn/scaler.pkl')
encoder = load('churn/encoder.pkl')

def home(request):
    return render(request, 'crm/home.html')

def model_performance(request):
    # Load sample data to evaluate the model
    data = pd.read_csv('Telco-Customer-Churn.csv')

    # Preprocess the data in the same way as training
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                   'PaperlessBilling', 'Churn']
    for col in binary_cols:
        data[col] = data[col].apply(lambda x: 1 if x == 'Yes' else 0)
    onehot_cols = ['InternetService', 'Contract', 'PaymentMethod']
    data = pd.get_dummies(data, columns=onehot_cols)
    data = data.drop(columns=['customerID'])

    # Define features and target
    X = data.drop('Churn', axis=1)
    y = data['Churn']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numeric data
    num_cols = X.select_dtypes(include=[np.float64, np.int64]).columns
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Transform categorical data
    cat_cols = X.select_dtypes(include=['uint8']).columns
    X_test_cat = encoder.transform(X_test[cat_cols])

    # Concatenate numeric and categorical data
    X_final = np.hstack((X_test[num_cols], X_test_cat.toarray()))

    # Predict with adjusted threshold
    y_pred_proba = pipeline.predict_proba(X_final)[:, 1]
    threshold = 0.4  # Adjusted threshold
    y_pred_adjusted = (y_pred_proba >= threshold).astype(int)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred_adjusted)
    precision = precision_score(y_test, y_pred_adjusted)
    recall = recall_score(y_test, y_pred_adjusted)
    conf_matrix = confusion_matrix(y_test, y_pred_adjusted)
    class_report = classification_report(y_test, y_pred_adjusted)

    context = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'conf_matrix': conf_matrix.tolist(),
        'class_report': class_report,
    }
    return render(request, 'crm/model_performance.html', context)

def churn_factors(request):
    # Display feature importance
    feature_names = load('churn/feature_names.pkl')
    feature_importance = load('churn/feature_importance.pkl')

    # Convert to a list of dictionaries for easier rendering
    feature_importance_list = feature_importance.to_dict(orient='records')

    context = {
        'feature_names': feature_names,
        'feature_importance': feature_importance_list,
    }
    return render(request, 'crm/churn_factors.html', context)



def at_risk_customers(request):
    # Load customer data
    data = pd.read_csv('Telco-Customer-Churn.csv')

    # Retain customerID before dropping it
    customer_ids = data['customerID']

    # Preprocess the data in the same way as training
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                   'PaperlessBilling', 'Churn']
    for col in binary_cols:
        data[col] = data[col].apply(lambda x: 1 if x == 'Yes' else 0)
    onehot_cols = ['InternetService', 'Contract', 'PaymentMethod']
    data = pd.get_dummies(data, columns=onehot_cols)

    # Define features
    X = data.drop(columns=['customerID', 'Churn'])

    # Scale numeric data
    num_cols = X.select_dtypes(include=[np.float64, np.int64]).columns
    X[num_cols] = scaler.transform(X[num_cols])

    # Transform categorical data
    cat_cols = X.select_dtypes(include=['uint8']).columns
    X_cat = encoder.transform(X[cat_cols])

    # Concatenate numeric and categorical data
    X_final = np.hstack((X[num_cols], X_cat.toarray()))

    # Predict churn probabilities
    y_pred_proba = pipeline.predict_proba(X_final)[:, 1]
    data['Churn_Probability'] = y_pred_proba

    # Include customerID for at-risk customers
    data['customerID'] = customer_ids

    # Filter out customers who have already churned
    at_risk_customers = data[data['Churn'] == 0]
    at_risk_customers = at_risk_customers[['customerID', 'Churn_Probability']]
    at_risk_customers = at_risk_customers.sort_values(by='Churn_Probability', ascending=False)

    context = {
        'at_risk_customers': at_risk_customers.to_dict(orient='records'),
    }
    return render(request, 'crm/at_risk_customers.html', context)

def model_metrics(request):
    # Load sample data to evaluate the model
    data = pd.read_csv('Telco-Customer-Churn.csv')

    # Preprocess the data in the same way as training
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                   'PaperlessBilling', 'Churn']
    for col in binary_cols:
        data[col] = data[col].apply(lambda x: 1 if x == 'Yes' else 0)
    onehot_cols = ['InternetService', 'Contract', 'PaymentMethod']
    data = pd.get_dummies(data, columns=onehot_cols)
    data = data.drop(columns=['customerID'])

    # Define features and target
    X = data.drop('Churn', axis=1)
    y = data['Churn']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numeric data
    num_cols = X.select_dtypes(include=[np.float64, np.int64]).columns
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Transform categorical data
    cat_cols = X.select_dtypes(include=['uint8']).columns
    X_test_cat = encoder.transform(X_test[cat_cols])

    # Concatenate numeric and categorical data
    X_final = np.hstack((X_test[num_cols], X_test_cat.toarray()))

    # Predict with adjusted threshold
    y_pred_proba = pipeline.predict_proba(X_final)[:, 1]
    threshold = 0.4  # Adjusted threshold
    y_pred_adjusted = (y_pred_proba >= threshold).astype(int)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred_adjusted)
    precision = precision_score(y_test, y_pred_adjusted)
    recall = recall_score(y_test, y_pred_adjusted)
    conf_matrix = confusion_matrix(y_test, y_pred_adjusted)
    class_report = classification_report(y_test, y_pred_adjusted)

    context = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'conf_matrix': conf_matrix.tolist(),
        'class_report': class_report,
    }
    return render(request, 'crm/model_metrics.html', context)

def retrain_model(request):
    # Load and preprocess data
    data = pd.read_csv('Telco-Customer-Churn.csv')
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                   'PaperlessBilling', 'Churn']
    for col in binary_cols:
        data[col] = data[col].apply(lambda x: 1 if x == 'Yes' else 0)
    onehot_cols = ['InternetService', 'Contract', 'PaymentMethod']
    data = pd.get_dummies(data, columns=onehot_cols)
    data = data.drop(columns=['customerID'])

    # Define features and target
    X = data.drop('Churn', axis=1)
    y = data['Churn']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Scale numeric data
    num_cols = X.select_dtypes(include=[np.float64, np.int64]).columns
    scaler = StandardScaler()
    X_train_resampled[num_cols] = scaler.fit_transform(X_train_resampled[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Transform categorical data
    cat_cols = X.select_dtypes(include=['uint8']).columns
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_train_cat = encoder.fit_transform(X_train_resampled[cat_cols])
    X_test_cat = encoder.transform(X_test[cat_cols])

    # Concatenate numeric and categorical data
    X_train_final = np.hstack((X_train_resampled[num_cols], X_train_cat.toarray()))
    X_test_final = np.hstack((X_test[num_cols], X_test_cat.toarray()))

    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_final, y_train_resampled)

    # Save the updated model, scaler, and encoder
    dump(model, 'churn/logistic_regression_model.pkl')
    dump(scaler, 'churn/scaler.pkl')
    dump(encoder, 'churn/encoder.pkl')

    return redirect('home')

