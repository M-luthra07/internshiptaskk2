# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 22:55:18 2025

@author: luthr
"""

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Title of the app
st.title("Linear Regression Web App")

# Load the dataset directly from a local file
DATA_PATH =  r"M-luthra07/internshiptaskk2/blob/main/parkinsons_updrs.data.csv"  # Replace with your dataset filename
try:
    data = pd.read_csv(DATA_PATH,encoding='ISO-8859-1')
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Select the feature and target columns
    feature_col = st.selectbox("Select the Feature Column", data.columns)
    target_col = st.selectbox("Select the Target Column", data.columns)

    if feature_col and target_col and feature_col != target_col:
        # Preparing data for the model
        X = data[[feature_col]]
        y = data[target_col]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Performance metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("### Model Performance Metrics")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"R-squared (R2): {r2:.2f}")

        # Plotting the regression line
        st.write("### Regression Line")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X, y, color="blue", alpha=0.5, label="Data Points")
        ax.plot(X, model.predict(X), color="red", label="Regression Line")
        ax.set_xlabel(feature_col)
        ax.set_ylabel(target_col)
        ax.set_title(f"Linear Regression: {feature_col} vs {target_col}")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Please select valid feature and target columns.")
except FileNotFoundError:
    st.error(f"The dataset file `{DATA_PATH}` was not found. Please ensure the file is in the same directory as this script.")
