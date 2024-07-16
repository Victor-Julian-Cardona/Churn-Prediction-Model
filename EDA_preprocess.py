import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
data = pd.read_csv('Telco-Customer-Churn.csv')

# Convert "Yes" and "No" to 1 and 0
binary_columns = [
    'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn'
]
for col in binary_columns:
    data[col] = data[col].apply(lambda x: 1 if x == 'Yes' else 0)

# Handle missing values in 'TotalCharges' and convert to float
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

# Drop CustomerID
data = data.drop('customerID', axis=1)

# Identify numerical and categorical columns
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = ['gender', 'InternetService', 'Contract', 'PaymentMethod', 
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(), cat_cols)
    ])

# Apply the preprocessing
X = data.drop('Churn', axis=1)
y = data['Churn']

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
X_preprocessed = pipeline.fit_transform(X)

# Convert preprocessed data back to DataFrame for inspection
X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=[
    'tenure', 'MonthlyCharges', 'TotalCharges'] + 
    list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
)

# Add 'Churn' column back to the preprocessed DataFrame for correlation analysis
X_preprocessed_df['Churn'] = y.values

# Plot distributions of numerical variables
num_features = len(num_cols) + len(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
fig, axes = plt.subplots(nrows=(num_features // 3) + 1, ncols=3, figsize=(20, 15))
axes = axes.flatten()

for i, col in enumerate(X_preprocessed_df.columns):
    ax = axes[i]
    ax.hist(X_preprocessed_df[col], bins=50)
    ax.set_title(f'Distribution of {col}', fontsize=12)

plt.tight_layout(pad=3.0)
plt.show()

# Correlation matrix
corr_matrix = X_preprocessed_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", annot_kws={"size": 8}, cmap='coolwarm')
plt.title('Correlation Matrix', fontsize=16)
plt.show()

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data['tenure'], data['MonthlyCharges'])
plt.xlabel('Tenure (Months)', fontsize=14)
plt.ylabel('Monthly Charges ($)', fontsize=14)
plt.title('Tenure vs. Monthly Charges', fontsize=16)
plt.show()

# Box plot
plt.figure(figsize=(10, 6))
data.boxplot(column='MonthlyCharges', by='Churn')
plt.xlabel('Churn', fontsize=14)
plt.ylabel('Monthly Charges ($)', fontsize=14)
plt.title('Monthly Charges by Churn', fontsize=16)
plt.suptitle('')  # Suppress the default title
plt.show()
