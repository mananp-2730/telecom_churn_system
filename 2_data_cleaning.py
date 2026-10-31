import os

import pandas as pd


def clean_data():
    print("Loading raw data...")
    # Load the CSV we just downloaded
    df = pd.read_csv(os.path.join("data", "raw_telecom_churn.csv"))
    
    print("Fixing 'TotalCharges' column...")
    # Force the column to numeric. 'coerce' turns blank spaces into NaN (Not a Number)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill those newly created NaN values with 0 (since they are brand new customers with 0 charges)
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    print("Converting 'Churn' to binary...")
    # Map 'Yes' to 1 and 'No' to 0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    print("Dropping 'customerID'...")
    # IDs are unique identifiers that add no predictive value to machine learning models
    df = df.drop(columns=['customerID'])
    
    # Save the cleaned data to a new file
    clean_file_path = os.path.join("data", "cleaned_telecom_churn.csv")
    df.to_csv(clean_file_path, index=False)
    
    print(f"\nSuccess! Cleaned data saved to {clean_file_path}")
    print(f"The dataset now has {df.shape[0]} rows and {df.shape[1]} columns.")
    print("\n--- First 3 rows of the cleaned 'TotalCharges' and 'Churn' columns ---")
    print(df[['TotalCharges', 'Churn']].head(3))

if __name__ == "__main__":
    clean_data()