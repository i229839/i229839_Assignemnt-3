import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Set Streamlit page config
st.set_page_config(page_title="Market Insights App", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stButton button {
            background-color: #6610f2;
            color: white;
            font-weight: bold;
        }
        h1, h2, h3 {
            color: #343a40;
        }
    </style>
""", unsafe_allow_html=True)

# App title and intro
st.title("📊 Market Insights App using Regression")
st.image("https://media.giphy.com/media/3oKIPtjElfqwMOTbH2/giphy.gif", width=400)
st.markdown("Explore historical stock prices, apply feature engineering, and train a regression model to gain insights.")

# Initialize session state variables
for key in ["data_loaded", "features_ready", "split_done", "model_trained"]:
    if key not in st.session_state:
        st.session_state[key] = False

for key in ["model", "X_train", "X_test", "y_train", "y_test"]:
    if key not in st.session_state:
        st.session_state[key] = None

if "data" not in st.session_state:
    st.session_state["data"] = pd.DataFrame()

# Sidebar for data input
st.sidebar.header("📥 Stock Data Input")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", "MSFT")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))

# Step 1: Load data
if st.sidebar.button("📡 Fetch Data"):
    data = yf.download(ticker, start=start_date, end=end_date)
    if not data.empty:
        st.success("✅ Stock data fetched successfully.")
        st.dataframe(data.tail())
        st.session_state.data = data
        st.session_state.data_loaded = True
        st.session_state.features_ready = False
        st.session_state.split_done = False
        st.session_state.model_trained = False
    else:
        st.error("❌ Could not load data. Please check the ticker or date range.")

# Step 2: Feature Engineering
if st.button("🧪 Feature Engineering"):
    if st.session_state.data_loaded:
        data = st.session_state.data.copy()
        adj_close_col = None
        for col in ["Adj Close", "AdjClose", "ADJ CLOSE", "Close"]:
            if col in data.columns:
                adj_close_col = col
                break

        if adj_close_col:
            data["Return"] = data[adj_close_col].pct_change()
            data["Lag1"] = data["Return"].shift(1)
            data.dropna(inplace=True)
            st.success("✅ Features engineered.")
            st.line_chart(data["Return"])
            st.session_state.data = data
            st.session_state.features_ready = True
        else:
            st.error("⚠️ Could not find 'Adj Close' or similar column in the dataset.")
    else:
        st.warning("⚠️ Please fetch the data first.")

# Step 3: Preprocessing
if st.button("🧹 Preprocessing"):
    if st.session_state.data_loaded:
        data = st.session_state.data.dropna()
        st.session_state.data = data
        st.success("✅ Preprocessing completed. NA values removed.")
        st.write(data.describe())
    else:
        st.warning("⚠️ Please fetch the data first.")

# Step 4: Train/Test Split
if st.button("🧪 Train/Test Split"):
    if st.session_state.features_ready:
        data = st.session_state.data
        X = data[["Lag1"]]
        y = data["Return"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.split_done = True

        st.success("✅ Data split into training and test sets.")
        fig = px.pie(values=[len(X_train), len(X_test)], names=["Train", "Test"], title="Train/Test Split")
        st.plotly_chart(fig)
    else:
        st.warning("⚠️ Please complete feature engineering first.")

# Step 5: Train Model
if st.button("🤖 Train Model"):
    if st.session_state.split_done:
        model = LinearRegression()
        model.fit(st.session_state.X_train, st.session_state.y_train)
        st.session_state.model = model
        st.session_state.model_trained = True
        st.success("✅ Linear Regression model trained.")
    else:
        st.warning("⚠️ Please split the data first.")

# Step 6: Evaluate Model
if st.button("📈 Evaluate Model"):
    if st.session_state.model_trained:
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        st.metric("Mean Squared Error", f"{mse:.6f}")
        st.metric("R² Score", f"{r2:.4f}")

        result_df = pd.DataFrame({"Actual": y_test.values, "Predicted": preds})
        st.line_chart(result_df)
    else:
        st.warning("⚠️ Train the model before evaluation.")

# Step 7: Visualize Predictions
if st.button("📊 Visualize Predictions"):
    if st.session_state.model_trained:
        results = pd.DataFrame({
            "Actual": st.session_state.y_test.values,
            "Predicted": st.session_state.model.predict(st.session_state.X_test)
        })
        fig = px.scatter(results, x="Actual", y="Predicted", title="📉 Actual vs Predicted Returns")
        st.plotly_chart(fig)
    else:
        st.warning("⚠️ Please train the model first.")
