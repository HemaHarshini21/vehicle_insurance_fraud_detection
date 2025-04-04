import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from joblib import dump

# Define the path to the CSV file
csv_path = r"C:\Users\91855\Documents\New folder\backend\carclaims.csv"

# Ensure the file exists
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"CSV file not found at: {csv_path}")

# Load the CSV file
print("Loading CSV file...")
df = pd.read_csv(csv_path)
print("CSV file loaded successfully.")

print("Current working directory:", os.getcwd())
# Check for the 'FraudFound' column
if 'FraudFound' not in df.columns:
    raise ValueError("Column 'FraudFound' is missing in the dataset.")

# Check for missing values in 'FraudFound'
if df['FraudFound'].isnull().any():
    print("Missing values detected in 'FraudFound'. Filling with default value 'No'.")
    df['FraudFound'] = df['FraudFound'].fillna('No')

# Map 'FraudFound' to binary values
df['FraudFound'] = df['FraudFound'].map({'No': 0, 'Yes': 1})

# Standardize categorical columns to lowercase and capitalize first letter
categorical_columns = df.select_dtypes(include=['object']).columns
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.str.strip().str.lower().str.capitalize())

# Encode categorical columns
print("Encoding categorical data...")
label_encoders = {}
for col in categorical_columns:
    print(f"Encoding column: {col}")
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
print("Categorical data encoded.")

# Split dataset into features and target
X = df.drop('FraudFound', axis=1)
y = df['FraudFound']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Dataset split into training and testing sets.")

# Train the XGBoost model
print("Training the XGBoost model...")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Define the path to save the model and encoders
model_save_path = "model.joblib"
encoders_save_path = "encoders.joblib"

# Save the model and encoders
print(f"Saving the model to {model_save_path}...")
dump(xgb_model, model_save_path)
print("Model saved successfully.")

print(f"Saving the encoders to {encoders_save_path}...")
dump(label_encoders, encoders_save_path)
print("Encoders saved successfully.")