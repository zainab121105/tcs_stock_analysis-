import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Title
st.title("TCS Stock Price Predictor")

# Load data
df = pd.read_csv("TCS_stock_history.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df.dropna(inplace=True)

# Feature columns
features = ['Open', 'High', 'Low', 'Volume', 'Day', 'Month', 'Year']
target = 'Close'

# Train model
X = df[features]
y = df[target]
model = LinearRegression()
model.fit(X, y)

# User input
st.sidebar.header("Input Stock Details")
open_price = st.sidebar.number_input("Open Price", min_value=0.0)
high_price = st.sidebar.number_input("High Price", min_value=0.0)
low_price = st.sidebar.number_input("Low Price", min_value=0.0)
volume = st.sidebar.number_input("Volume", min_value=0)
day = st.sidebar.number_input("Day", min_value=1, max_value=31)
month = st.sidebar.number_input("Month", min_value=1, max_value=12)
year = st.sidebar.number_input("Year", min_value=2000, max_value=2100)

# Predict
if st.sidebar.button("Predict Closing Price"):
    input_data = pd.DataFrame([{
        'Open': open_price,
        'High': high_price,
        'Low': low_price,
        'Volume': volume,
        'Day': day,
        'Month': month,
        'Year': year
    }])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Closing Price: ₹{prediction:.2f}")
