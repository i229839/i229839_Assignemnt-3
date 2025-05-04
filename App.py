import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Set Streamlit page config
st.set_page_config(page_title="Finance ML App", layout="centered")

# Custom CSS for background and buttons
st.markdown("""
    <style>
        .main { background-color: #f0f2f6; }
        .stButton button { background-color: #004080; color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Financial ML App with Linear Regression")

# Initialize session state variables
if "data_loaded" not in st.session_state:
    st.session_state["data_loaded"] = False
if "features_ready" not in st.session_state:
    st.session_state["features_ready"] = False
if "split_done" not in st.session_state:
    st.session_state["split_done"] = False
if "model_trained" not in st.session_state:
    st.session_state["model_trained"] = False
if "model" not in st.session_state:
    st.session_state["model"] = None
if "X_train" not in st.session_state:
    st.session_state["X_train"] = None
if "X_test" not in st.session_state:
    st.session_state["X_test"] = None
if "y_train" not in st.session_state:
    st.session_state["y_train"] = None
if "y_test" not in st.session_state:
    st.session_state["y_test"] = None
if "data" not in st.session_state:
    st.session_state["data"] = pd.DataFrame()

# Welcome message
st.markdown("### Welcome to the Streamlit ML App!")
st.image("https://media.giphy.com/media/3o6gbbuLW76jkt8vIc/giphy.gif", width=400)
st.markdown("Use the sidebar to get started by fetching stock data.")

# Sidebar for input
st.sidebar.header("📊 Data Input")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))

# Step buttons
if st.sidebar.button("Fetch Yahoo Finance Data"):
    data = yf.download(ticker, start=start_date, end=end_date)
    if not data.empty:
        st.success("✅ Data Loaded Successfully!")
        st.dataframe(data.tail())
        st.session_state["data"] = data  # Store the data in session state
        st.session_state["data_loaded"] = True
        st.session_state["features_ready"] = False # Reset subsequent steps
        st.session_state["split_done"] = False
        st.session_state["model_trained"] = False
        st.session_state["model"] = None
    else:
        st.error("❌ Failed to load data. Check ticker or date range.")

# Continue only if data is loaded
if st.button("Feature Engineering"):
    if st.session_state["data_loaded"] and not st.session_state["data"].empty:
        data = st.session_state["data"].copy()
        adj_close_col = None
        possible_cols = ["Adj Close", "AdjClose", "ADJ CLOSE", "Close"] # Add other potential names

        for col in possible_cols:
            if col in data.columns:
                adj_close_col = col
                break

        if adj_close_col:
            data["Return"] = data[adj_close_col].pct_change()
            data["Lag1"] = data["Return"].shift(1)
            data = data.dropna()
            st.success("✅ Features Created.")
            st.line_chart(data["Return"])
            st.session_state["data"] = data # Update data with features
            st.session_state["features_ready"] = True
            st.session_state["split_done"] = False # Reset next steps
            st.session_state["model_trained"] = False
            st.session_state["model"] = None
        else:
            st.error(f"❌ Could not find 'Adj Close' or similar column in the downloaded data. Available columns: {data.columns.tolist()}")
    else:
        st.warning("⚠️ Please fetch the stock data first.")

# Preprocessing
if st.button("Preprocessing"):
    if st.session_state["data_loaded"] and not st.session_state["data"].empty:
        data = st.session_state["data"].dropna()
        st.success("✅ Missing values removed.")
        st.write(data.describe())
        st.session_state["data"] = data # Update processed data
    else:
        st.warning("⚠️ Please fetch the stock data first.")

# Train/Test Split
if st.button("Train/Test Split"):
    if st.session_state["features_ready"] and not st.session_state["data"].empty:
        data = st.session_state["data"]
        X = data[["Lag1"]]
        y = data["Return"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.success("✅ Data Split into Training and Test Sets.")
        fig = px.pie(values=[len(X_train), len(X_test)], names=["Train", "Test"], title="Data Split")
        st.plotly_chart(fig)
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test
        st.session_state["split_done"] = True
        st.session_state["model_trained"] = False # Reset next step
        st.session_state["model"] = None
    else:
        st.warning("⚠️ Please fetch data and perform feature engineering first.")

# Model Training
if st.button("Train Model"):
    if st.session_state["split_done"]:
        model = LinearRegression()
        model.fit(st.session_state["X_train"], st.session_state["y_train"])
        st.success("✅ Model Trained Successfully.")
        st.session_state["model"] = model
        st.session_state["model_trained"] = True
    else:
        st.warning("⚠️ Please perform the train/test split first.")

# Evaluation
if st.button("Evaluate Model"):
    if st.session_state["model_trained"]:
        preds = st.session_state["model"].predict(st.session_state["X_test"])
        mse = mean_squared_error(st.session_state["y_test"], preds)
        r2 = r2_score(st.session_state["y_test"], preds)
        st.write(f"**Mean Squared Error:** {mse:.6f}")
        st.write(f"**R² Score:** {r2:.2f}")
        results_df = pd.DataFrame({"Actual": st.session_state["y_test"].values, "Predicted": preds})
        st.line_chart(results_df)
    else:
        st.warning("⚠️ Please train the model first by clicking the 'Train Model' button.")

# Results Visualization
if st.button("Visualize Predictions"):
    if st.session_state["model_trained"]:
        results = pd.DataFrame({
            "Actual": st.session_state["y_test"].values,
            "Predicted": st.session_state["model"].predict(st.session_state["X_test"])
        })
        fig = px.scatter(results, x="Actual", y="Predicted", title="Actual vs. Predicted Returns")
        st.plotly_chart(fig)
    else:
        st.warning("⚠️ Please train the model first by clicking the 'Train Model' button.")