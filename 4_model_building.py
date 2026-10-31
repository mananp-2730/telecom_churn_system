import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def build_model():
    print("Loading cleaned data...")
    df = pd.read_csv(os.path.join("data", "cleaned_telecom_churn.csv"))
    
    print("Applying One-Hot Encoding to categorical text columns...")
    # This converts text categories into 1/0 numeric columns
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    print(f"Data shape after encoding: {df_encoded.shape[0]} rows, {df_encoded.shape[1]} columns")
    
    # Separate the Features (X) from the Target (y)
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    
    print("Splitting data into 80% training and 20% testing...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training the Random Forest model (this may take a few seconds)...")
    # Initialize the model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train (fit) the model using the 80% training data
    rf_model.fit(X_train, y_train)
    
    print("Testing the model on the unseen 20% data...")
    # Ask the model to predict churn on the test data
    predictions = rf_model.predict(X_test)
    
    # Evaluate how well it did
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n--- Model Performance ---")  # noqa: F541
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
    print("Detailed Classification Report:")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    build_model()