import os

import pandas as pd


def load_and_save_data():
    print("Downloading dataset...")
    # The direct link to the standard IBM Telco Churn dataset
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    
    # Read the data from the internet
    df = pd.read_csv(url)
    
    # Save a local copy in your data folder
    file_path = os.path.join("data", "raw_telecom_churn.csv")
    df.to_csv(file_path, index=False)
    
    print(f"Data successfully saved to {file_path}!")
    print("\n--- Here are the first 5 rows of your data ---")
    
    # Display all columns cleanly
    pd.set_option('display.max_columns', None)
    print(df.head())

if __name__ == "__main__":
    load_and_save_data()