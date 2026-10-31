import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def generate_retention_offers():
    print("Loading data and preparing the model...")
    # Load original data for reference and encoded data for the model
    df_original = pd.read_csv(os.path.join("data", "cleaned_telecom_churn.csv"))
    df_encoded = pd.get_dummies(df_original, drop_first=True)
    
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    
    # Train the model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    print("Calculating Churn Risk Scores for all customers...")
    # predict_proba returns the probability of [Class 0 (Stay), Class 1 (Churn)]
    # We slice [:, 1] to grab just the probability of churning
    churn_probabilities = rf_model.predict_proba(X)[:, 1]
    
    # Attach the probabilities back to a fresh dataframe for our business team
    business_df = df_original[['tenure', 'MonthlyCharges', 'Contract']].copy()
    business_df['Churn_Risk_Score'] = churn_probabilities
    
    # Define High Value threshold (e.g., spending more than the median customer)
    median_spend = business_df['MonthlyCharges'].median()
    
    print("Assigning segment-wise retention offers...")
    def assign_offer(row):
        risk = row['Churn_Risk_Score']
        value = row['MonthlyCharges']
        
        if risk >= 0.65 and value >= median_spend:
            return "Tier 1: High Touch - Offer 12-month lock-in discount & free premium support."
        elif risk >= 0.65 and value < median_spend:
            return "Tier 2: Down-sell - Offer lighter data bundle or prepaid incentives."
        elif 0.35 <= risk < 0.65 and value >= median_spend:
            return "Tier 3: Proactive - Offer auto-pay setup bonus ($10 bill credit)."
        else:
            return "Tier 4: Low Risk - No retention spend. Target for up-selling."

    # Apply the logic to every customer
    business_df['Retention_Action'] = business_df.apply(assign_offer, axis=1)
    
    # Save the final actionable list
    output_path = os.path.join("data", "targeted_retention_campaign.csv")
    business_df.to_csv(output_path, index=False)
    
    print(f"\nSuccess! Targeted campaign list saved to {output_path}")
    print("\n--- Campaign Distribution ---")
    print(business_df['Retention_Action'].value_counts())
    print("\n--- Preview of High-Risk Customers ---")
    print(business_df[business_df['Churn_Risk_Score'] >= 0.65].head(3))

if __name__ == "__main__":
    generate_retention_offers()