import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


def plot_feature_importance():
    print("Loading and preparing data...")
    df = pd.read_csv(os.path.join("data", "cleaned_telecom_churn.csv"))
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    
    print("Training Random Forest to extract feature importance...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    # We train on the whole dataset this time to get the most accurate driver list
    rf_model.fit(X, y)
    
    print("Calculating top 10 churn drivers...")
    # Extract the importance scores and map them to column names
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    })
    
    # Sort from highest to lowest and grab the top 10
    importances = importances.sort_values(by='Importance', ascending=False).head(10)
    
    print("Generating visual...")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importances, x='Importance', y='Feature', palette='magma')
    plt.title('Top 10 Drivers of Customer Churn')
    plt.xlabel('Relative Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Save to our existing visuals folder
    save_path = os.path.join("visuals", "feature_importance.png")
    plt.savefig(save_path)
    plt.close()
    
    print(f"Success! Feature importance chart saved to {save_path}")
    print("\n--- Top 3 Drivers ---")
    for index, row in importances.head(3).iterrows():
        print(f"- {row['Feature']}: {row['Importance']:.4f}")

if __name__ == "__main__":
    plot_feature_importance()