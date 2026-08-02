import streamlit as st
import pandas as pd
import joblib
import datetime

# -------------------------
# Page Configuration
# -------------------------

st.set_page_config(page_title="Sales Forecasting System", page_icon="📈", layout="wide")

# -------------------------
# Load Model
# -------------------------

model = joblib.load(
    r"C:\Users\shubh\OneDrive\Documents\sales analysis and forecasting\model\sales_forecasting_new_model.pkl",
    mmap_mode="r",
)

# -------------------------
# Title
# -------------------------

st.title("📈 Sales Forecasting System")

st.markdown("""
Predict future store sales using Machine Learning.

Fill in the store details and click **Predict Sales**.
""")

# -------------------------
# Sidebar
# -------------------------


st.sidebar.header("Store Information")


store = st.sidebar.number_input("Store", 1, 1115, 1)

day_of_week = st.sidebar.selectbox("Day Of Week", list(range(7)))

customers = st.sidebar.number_input("Customers", 0, 10000, 500)

open_store = st.sidebar.selectbox("Store Open", [0, 1])

promo = st.sidebar.selectbox("Promo", [0, 1])

schoolholiday = st.sidebar.selectbox("School Holiday", [0, 1])


year = st.sidebar.number_input("Year", 2013, 2030, 2015)
month = st.sidebar.selectbox("Month", list(range(1, 13)))
day = st.sidebar.slider("Day", 1, 31, 1)
week = st.sidebar.number_input("Week", 1, 53, 1)
quarter = st.sidebar.selectbox("Quarter", [1, 2, 3, 4])

is_weekend = st.sidebar.selectbox("Weekend", [0, 1])
is_month_start = st.sidebar.selectbox("Month Start", [0, 1])
is_month_end = st.sidebar.selectbox("Month End", [0, 1])

stateholiday_a = st.sidebar.selectbox("State Holiday A", [0, 1])
stateholiday_b = st.sidebar.selectbox("State Holiday B", [0, 1])
stateholiday_c = st.sidebar.selectbox("State Holiday C", [0, 1])

lag1 = st.sidebar.number_input("Previous Day Sales", value=5000.0)
lag7 = st.sidebar.number_input("Sales 7 Days Ago", value=5100.0)
lag30 = st.sidebar.number_input("Sales 30 Days Ago", value=5200.0)


rollingmean7 = st.sidebar.number_input("Rolling Mean (7)", value=5050.0)
rollingmean30 = st.sidebar.number_input("Rolling Mean (30)", value=5000.0)
rollingstd7 = st.sidebar.number_input("Rolling Std (7)", value=200.0)


expandingmean = st.sidebar.number_input("Historical Average Sales", value=4900.0)

rollingstd30 = st.sidebar.number_input("Rolling Std (30)", value=250.0)


input_df = pd.DataFrame(
    {
        "Store": [store],
        "DayOfWeek": [day_of_week],
        "Customers": [customers],
        "Open": [open_store],
        "Promo": [promo],
        "SchoolHoliday": [schoolholiday],
        "Year": [year],
        "Month": [month],
        "Day": [day],
        "Week": [week],
        "Quarter": [quarter],
        "IsWeekend": [is_weekend],
        "IsMonthStart": [is_month_start],
        "IsMonthEnd": [is_month_end],
        "StateHoliday_a": [stateholiday_a],
        "StateHoliday_b": [stateholiday_b],
        "StateHoliday_c": [stateholiday_c],
        "Lag_1": [lag1],
        "Lag_7": [lag7],
        "Lag_30": [lag30],
        "RollingMean7": [rollingmean7],
        "RollingMean30": [rollingmean30],
        "RollingStd7": [rollingstd7],
        "ExpandingMean": [expandingmean],
        "RollingStd30": [rollingstd30],
    }
)


# -------------------------
# Prediction
# -------------------------)
if st.button("Predict Sales"):

    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Sales: ₹ {prediction:,.2f}")


# -------------------------
# Footer
# -------------------------

st.markdown("---")
st.write("Developed by **Shubhangi Singh**")
