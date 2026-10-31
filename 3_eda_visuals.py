import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_visuals():
    print("Loading cleaned data...")
    df = pd.read_csv(os.path.join("data", "cleaned_telecom_churn.csv"))
    
    # Create a new folder specifically for our charts
    os.makedirs("visuals", exist_ok=True)
    
    print("Generating 'Churn by Contract Type' chart...")
    plt.figure(figsize=(8, 5))
    # Countplot shows the number of customers in each contract type, split by whether they churned
    sns.countplot(data=df, x='Contract', hue='Churn', palette='Set2')
    plt.title('Churn by Contract Type (0 = Retained, 1 = Churned)')
    plt.ylabel('Number of Customers')
    plt.savefig(os.path.join("visuals", "churn_by_contract.png"))
    plt.close()
    
    print("Generating 'Tenure Distribution by Churn' chart...")
    plt.figure(figsize=(8, 5))
    # Histplot shows how long customers stay, colored by churn status
    sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', palette='Set2', bins=36)
    plt.title('Customer Tenure Distribution by Churn Status')
    plt.xlabel('Tenure (Months)')
    plt.ylabel('Number of Customers')
    plt.savefig(os.path.join("visuals", "churn_by_tenure.png"))
    plt.close()
    
    print("\nSuccess! Visualizations saved in the 'visuals' folder.")

if __name__ == "__main__":
    create_visuals()