import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# App title and description
st.title("Linear Regression App")
st.write("""
This app lets you:
1. Upload your dataset.
2. Choose the feature(s) and target for linear regression.
3. Train the model and visualize predictions.
""")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    # Load the data
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())
    
    # Step 2: Select Features and Target
    st.write("### Select Features and Target for Linear Regression")
    features = st.multiselect("Choose feature columns (independent variables):", options=data.columns)
    target = st.selectbox("Choose target column (dependent variable):", options=data.columns)
    
    if features and target:
        X = data[features]
        y = data[target]
        
        # Step 3: Split Data
        st.write("### Splitting Data")
        test_size = st.slider("Test Set Size (as a fraction of total data):", 0.1, 0.5, 0.2, 0.01)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        st.write(f"Training Data Size: {len(X_train)}, Test Data Size: {len(X_test)}")
        
        # Step 4: Train Linear Regression Model
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Model Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write("### Model Metrics")
        st.write(f"Mean Squared Error (MSE): {mse:.4f}")
        st.write(f"R² Score: {r2:.4f}")
        
        # Step 5: Visualization
        st.write("### Visualization")
        if len(features) == 1:
            # Single feature: Scatter plot with regression line
            plt.figure(figsize=(10, 6))
            plt.scatter(X_test, y_test, label="Actual Data", color="blue")
            plt.plot(X_test, y_pred, label="Predicted Line", color="red", linewidth=2)
            plt.xlabel(features[0])
            plt.ylabel(target)
            plt.legend()
            st.pyplot(plt)
        else:
            # Multiple features: Display predictions
            st.write("Predictions for the test set:")
            st.write(pd.DataFrame({"Actual": y_test, "Predicted": y_pred}))
