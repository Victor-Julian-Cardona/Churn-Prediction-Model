import os
import django
import pandas as pd
import sys

# Ensure the project root is in the Python path
sys.path.append('C:/Users/victo/Documents/Churn-Prediction-Model')

# Set the Django settings module environment variable
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'churn_project.settings')

# Initialize Django
django.setup()

# Import the Customer model from the churn app
from churn.models import Customer

def load_data():
    # Read the CSV file into a DataFrame
    data = pd.read_csv('Telco-Customer-Churn.csv')
    
    # Replace spaces with NaN
    data['TotalCharges'] = data['TotalCharges'].replace(" ", pd.NA)
    
    # Convert the TotalCharges column to float after ensuring all values are clean
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    
    # Fill NaN values with the median
    data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

    # Iterate over the DataFrame and create Customer objects
    for index, row in data.iterrows():
        Customer.objects.create(
            customerID=row['customerID'],
            gender=row['gender'],
            SeniorCitizen=row['SeniorCitizen'],
            Partner=row['Partner'],
            Dependents=row['Dependents'],
            tenure=row['tenure'],
            PhoneService=row['PhoneService'],
            MultipleLines=row['MultipleLines'],
            InternetService=row['InternetService'],
            OnlineSecurity=row['OnlineSecurity'],
            OnlineBackup=row['OnlineBackup'],
            DeviceProtection=row['DeviceProtection'],
            TechSupport=row['TechSupport'],
            StreamingTV=row['StreamingTV'],
            StreamingMovies=row['StreamingMovies'],
            Contract=row['Contract'],
            PaperlessBilling=row['PaperlessBilling'],
            PaymentMethod=row['PaymentMethod'],
            MonthlyCharges=row['MonthlyCharges'],
            TotalCharges=row['TotalCharges'],
            Churn=row['Churn']
        )

if __name__ == "__main__":
    load_data()
