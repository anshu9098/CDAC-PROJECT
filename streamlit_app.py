
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from lifetimes import ParetoNBDFitter
from lifetimes.plotting import plot_period_transactions

# Set the title and sidebar information
st.title("CLV Prediction and Segmentation App")
st.sidebar.title("Input Features :pencil:")

# Upload CSV file
data = st.file_uploader("Upload your customer data (CSV)", type=["csv"])

# Sidebar image and project info
st.sidebar.image("https://www.adlibweb.com/wp-content/uploads/2020/06/customer-lifetime-value.jpg", width=150)
st.sidebar.markdown("**CDAC Project**")

# Sidebar sliders for days and profit margin
days = st.sidebar.slider("Select the number of days", min_value=30, max_value=365, value=180, step=30)
profit_margin = st.sidebar.slider("Select the profit margin", min_value=0.01, max_value=1.0, value=0.1, step=0.01)

# Processing uploaded data
if data is not None:
    df = pd.read_csv(data)
    
    # Data preparation
    df['monetary_value'] = df['monetary_value'] * profit_margin

    # Apply KMeans Clustering
    kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=1000, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['frequency', 'recency', 'monetary_value']])
    
    # Display clustering results
    st.subheader("Customer Segmentation Clustering")
    cluster_counts = df['Cluster'].value_counts()
    st.write(cluster_counts)

    # Visualizing the clusters
    st.subheader("Cluster Distribution")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Cluster', data=df)
    st.pyplot(plt)
    
    # Displaying Customer Lifetime Value Prediction
    st.subheader("Customer Lifetime Value (CLV) Prediction")
    clv_summary = df.groupby('Cluster').agg({
        'predicted_clv_p': 'mean',
        'profit_margin': 'mean'
    })
    st.write(clv_summary)
    
    # Visualizing CLV distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y='predicted_clv_p', data=df)
    plt.title('Predicted CLV Distribution by Cluster')
    st.pyplot(plt)

    # Customer Level Analysis
    st.subheader("Individual Customer Analysis")
    customer_id = st.selectbox("Select Customer ID", df['CustomerID'].unique())
    customer_data = df[df['CustomerID'] == customer_id]
    st.write(customer_data)

    # Pareto/NBD Model for Customer Lifetime Prediction
    st.subheader("Pareto/NBD Model")
    pareto_model = ParetoNBDFitter(penalizer_coef=0.0)
    pareto_model.fit(df['frequency'], df['recency'], days, df['monetary_value'])
    plot_period_transactions(pareto_model)
    st.pyplot(plt)

else:
    st.warning("Please upload a CSV file to proceed.")
