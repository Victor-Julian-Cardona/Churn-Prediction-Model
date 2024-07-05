import os
import django
import pandas as pd

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'churn_project.settings')
django.setup()

from churn.models import Customer

def load_data():
    data = pd.read_csv('../Telco-Customer-Churn.csv')
    data['TotalCharges'] = data['TotalCharges'].replace(" ", pd.NA)
    data['TotalCharges'] = data['TotalCharges'].astype(float)
    data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

    # Load data into Django models
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
