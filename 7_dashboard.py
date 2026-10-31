import os

import pandas as pd
import streamlit as st
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Churn Retention Engine", page_icon="🔄", layout="wide")

st.title("Telecom Customer Churn & Retention Engine")
st.markdown("Analyze churn drivers, interpret model decisions, and deploy segment-wise retention offers.")

# 2. Load the Final Campaign Data
@st.cache_data
def load_campaign_data():
    file_path = os.path.join("data", "targeted_retention_campaign.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame() # Return empty if not found

df_campaign = load_campaign_data()

# 3. Create Dashboard Tabs
tab1, tab2, tab3 = st.tabs(["Exploratory Analysis", "Model Insights", "Retention Strategy"])

# --- TAB 1: EDA ---
with tab1:
    st.header("Friction Points: Contract Type & Tenure")
    st.markdown("Visualizing historical customer behavior to identify immediate risk segments.")
    
    col1, col2 = st.columns(2)
    with col1:
        contract_img = os.path.join("visuals", "churn_by_contract.png")
        if os.path.exists(contract_img):
            st.image(Image.open(contract_img), use_container_width=True)
    with col2:
        tenure_img = os.path.join("visuals", "churn_by_tenure.png")
        if os.path.exists(tenure_img):
            st.image(Image.open(tenure_img), use_container_width=True)

# --- TAB 2: MODELING ---
with tab2:
    st.header("Random Forest Feature Importance")
    st.markdown("**Model Performance Baseline:** ~78.5% Accuracy.")
    st.markdown("These are the mathematical drivers the algorithm relies on to flag a customer as high-risk.")
    
    feat_img = os.path.join("visuals", "feature_importance.png")
    if os.path.exists(feat_img):
        st.image(Image.open(feat_img), width=800)

# --- TAB 3: RETENTION ENGINE ---
with tab3:
    st.header("Actionable Retention Engine")
    st.markdown("Filter the customer base by assigned retention tier to deploy targeted campaigns.")
    
    if not df_campaign.empty:
        # Create an interactive dropdown filter
        tiers = sorted(df_campaign['Retention_Action'].unique())
        selected_tier = st.selectbox("Select Campaign Tier:", tiers)
        
        # Filter the dataframe based on selection
        filtered_df = df_campaign[df_campaign['Retention_Action'] == selected_tier]
        
        # Display high-level metrics for the selection
        col1, col2 = st.columns(2)
        col1.metric("Customers in Segment", len(filtered_df))
        col2.metric("Average Risk Score", f"{filtered_df['Churn_Risk_Score'].mean():.2f}")
        
        # Display the interactive dataframe
        st.dataframe(filtered_df.sort_values(by='Churn_Risk_Score', ascending=False), use_container_width=True)
    else:
        st.warning("Campaign data not found. Please run the retention engine script first.")