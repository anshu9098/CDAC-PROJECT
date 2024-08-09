
# Adding necessary libraries
import streamlit as st
import pandas as pd
import lifetimes
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Configurations
np.random.seed(42)
warnings.filterwarnings("ignore")

# App title
st.markdown(""" # CLV Prediction and Segmentation App """)

# App description and file uploader
st.image("https://ultracommerce.co/wp-content/uploads/2022/04/maximize-customer-lifetime-value.png", use_column_width=True)
data = st.file_uploader("Upload your RFM data (CSV format only)", type="csv")

# Sidebar settings
st.sidebar.image("https://www.adlibweb.com/wp-content/uploads/2020/06/customer-lifetime-value.jpg", width=150)
st.sidebar.markdown("**CDAC Project**")
st.sidebar.title("Input Features :pencil:")

days = st.sidebar.slider("Select The No. Of Days", min_value=1, max_value=365, step=1)
profit = st.sidebar.slider("Select the Profit Margin", min_value=0.01, max_value=0.09, step=0.01)

st.sidebar.markdown("""
Before uploading the file, please select the input features first.
Make sure the columns are in the proper format. You can download the [dummy data](https://github.com/tejas-tilekar/CDAC-Project/blob/main/sample_file.csv) for reference.
**Note:** Only Use "CSV" File.
""")

# Function to load data and apply models
def load_data(data, days, profit):
    input_data = pd.read_csv(data)
    input_data = pd.DataFrame(input_data.iloc[:, 1:])
    
    # Pareto-NBD Model
    pareto_model = lifetimes.ParetoNBDFitter(penalizer_coef=0.0)
    pareto_model.fit(input_data["frequency"], input_data["recency"], input_data["T"])
    
    # Predict future purchases using the selected number of days
    input_data["predicted_purchases"] = pareto_model.conditional_expected_number_of_purchases_up_to_time(
        days, input_data["frequency"], input_data["recency"], input_data["T"]
    )
    
    # Gamma-Gamma Model
    input_data = input_data[(input_data["frequency"] > 0) & (input_data["monetary_value"] > 0)]
    input_data.reset_index(drop=True, inplace=True)
    
    ggf_model = lifetimes.GammaGammaFitter(penalizer_coef=0.0)
    ggf_model.fit(input_data["frequency"], input_data["monetary_value"])
    
    # Calculate CLV using the selected number of days
    input_data["predicted_clv"] = ggf_model.customer_lifetime_value(
        pareto_model, input_data["frequency"], input_data["recency"],
        input_data["T"], input_data["monetary_value"], time=days, freq='D', discount_rate=0.01
    )
    
    # Apply the profit margin
    input_data["profit_margin"] = input_data["predicted_clv"] * profit
    
    # K-Means Clustering
    col = ["predicted_purchases", "predicted_clv", "profit_margin"]
    new_df = input_data[col]
    
    k_model = KMeans(n_clusters=5, init="k-means++", max_iter=1000)
    k_model_fit = k_model.fit(new_df)
    
    input_data["Labels"] = k_model_fit.labels_
    
    return input_data

# Main app logic
if data is not None:
    st.write("Calculating CLV and segmenting customers...")
    input_data = load_data(data, days, profit)
    st.write(input_data)
    
    # Display the clustered data or any specific visualization
    st.write("CLV Distribution per Cluster:")
    fig, ax = plt.subplots()
    sns.boxplot(x="Labels", y="predicted_clv", data=input_data, ax=ax)
    st.pyplot(fig)

    # Debugging outputs
    st.write(f"Selected Days: {days}")
    st.write(f"Selected Profit Margin: {profit}")
    st.write("Updated CLV and Profit Margin calculations:")
    st.write(input_data[["predicted_clv", "profit_margin"]].head())
