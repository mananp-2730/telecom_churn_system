# Telecom Customer Churn & Retention Engine

## Project Overview
This project tackles a core business analytics problem: telecom companies lose 15–25% of their subscribers annually, and customer acquisition costs far outweigh retention costs. This system analyzes billing, tenure, and usage data to identify churn drivers, calculate individual churn risk scores, and assign segment-wise retention offers.

## Features
* **Exploratory Data Analysis:** Visualizes historical churn drivers (Contract Type, Tenure).
* **Predictive Modeling:** Utilizes a Random Forest Classifier to achieve a ~78.5% baseline accuracy in predicting churn.
* **Feature Importance:** Identifies `TotalCharges`, `tenure`, and `MonthlyCharges` as the top indicators of customer churn.
* **Retention Strategy Engine:** Segments customers into an actionable 4-tier matrix based on predicted risk score and monthly recurring revenue (ARPU).
* **Interactive Dashboard:** A local Streamlit web application for stakeholders to filter and explore targeted retention campaigns.

## Tech Stack
* **Language:** Python
* **Libraries:** pandas, scikit-learn, matplotlib, seaborn, streamlit
* **Methodology:** CRISP-DM

## How to Run Locally
1. Clone the repository.
2. Create and activate a virtual environment: `python -m venv env`
3. Install dependencies: `pip install -r requirements.txt`
4. Execute the pipeline scripts sequentially (`1_data_pipeline.py` through `6_retention_engine.py`).
5. Launch the dashboard: `streamlit run 7_dashboard.py`