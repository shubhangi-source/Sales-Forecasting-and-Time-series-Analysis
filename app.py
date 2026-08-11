import streamlit as st
import pandas as pd
import joblib
import datetime
import os
import gdown
import numpy as np


# -------------------------
# Page Configuration
# -------------------------

st.set_page_config(page_title="Sales Forecasting System", page_icon="📈", layout="wide")

os.makedirs("model", exist_ok=True)
model_path = "model/sales_forecasting_new_model.pkl"

# Download model from Google Drive if it isn't already present
if not os.path.exists(model_path):
    file_id = "1sEDTak7AphaBtoT-nu0ujF_u-iv9Aei8"
    url = f"https://drive.google.com/uc?id={file_id}"
    # Replace line 22:
    file_id = "1sEDTak7AphaBtoT-nu0ujF_u-iv9Aei8"
    gdown.download(id=file_id, output=model_path, quiet=False)

# Load the model
model = joblib.load(model_path)


tab1, tab2, tab3, tab4 = st.tabs(
    ["🏠 Home", "📈 Prediction", "📊 Analytics", "ℹ️ About"]
)

with tab1:

    st.title("📈 AI Sales Forecasting Dashboard")

    st.write("""
    Welcome to the Sales Forecasting System.

    This application predicts future sales using a Machine Learning model
    trained on the Rossmann Store Sales dataset.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Stores", "1115")
    col2.metric("Features", "20+")
    col3.metric("Model", "Random Forest")

    st.divider()

    st.subheader("Project Highlights")

    st.markdown("""
    ✅ Data Cleaning

    ✅ Feature Engineering

    ✅ Machine Learning Model

    ✅ Sales Prediction

    ✅ Interactive Dashboard
    """)


with tab2:

    st.header("Predict Future Sales")

    col1, col2, col3 = st.columns(3)

    with col1:

        store = st.number_input("Store", 1, 1115, 1)

        day_of_week = st.selectbox("Day Of Week", list(range(7)))

        customers = st.number_input("Customers", 0, 10000, 500)

        open_store = st.selectbox("Store Open", [0, 1])

        promo = st.selectbox("Promo", [0, 1])

        schoolholiday = st.selectbox("School Holiday", [0, 1])

        year = st.number_input("Year", 2013, 2030, 2015)
        month = st.selectbox("Month", list(range(1, 13)))
        day = st.slider("Day", 1, 31, 1)

    with col2:

        week = st.number_input("Week", 1, 53, 1)
        quarter = st.selectbox("Quarter", [1, 2, 3, 4])

        is_weekend = st.selectbox("Weekend", [0, 1])
        is_month_start = st.selectbox("Month Start", [0, 1])
        is_month_end = st.selectbox("Month End", [0, 1])
        stateholiday_a = st.selectbox("State Holiday A", [0, 1])
        stateholiday_b = st.selectbox("State Holiday B", [0, 1])
        stateholiday_c = st.selectbox("State Holiday C", [0, 1])

        lag1 = st.number_input("Previous Day Sales", value=5000.0)

    with col3:

        lag7 = st.number_input("Sales 7 Days Ago", value=5100.0)
        lag30 = st.number_input("Sales 30 Days Ago", value=5200.0)

        rollingmean7 = st.number_input("Rolling Mean (7)", value=5050.0)
        rollingmean30 = st.number_input("Rolling Mean (30)", value=5000.0)
        rollingstd7 = st.number_input("Rolling Std (7)", value=200.0)

        expandingmean = st.number_input("Historical Average Sales", value=4900.0)

        rollingstd30 = st.number_input("Rolling Std (30)", value=250.0)
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


import streamlit as st
import plotly.express as px

with tab3:

    st.header("📊 Sales Analytics Dashboard")
    st.markdown("Explore historical sales patterns and business insights.")

    # -------------------------
    # KPI Cards
    # -------------------------

    # Download CSV from Google Drive if not present
    # Download CSV from Google Drive if not present
    csv_path = "data/featured__new_sales_data.csv"
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(csv_path):
        csv_file_id = "1N8C_0E2WJaJg30VICuIXcqTFFY3qobRK"
        csv_url = f"https://drive.google.com/uc?id={csv_file_id}"
        gdown.download(csv_url, csv_path, quiet=False)

    df = pd.read_csv(csv_path)
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("💰 Total Sales", f"₹ {df['Sales'].sum():,.0f}")

    c2.metric("📈 Average Sales", f"₹ {df['Sales'].mean():,.0f}")

    c3.metric("👥 Total Customers", f"{df['Customers'].sum():,.0f}")

    c4.metric("🏪 Total Stores", df["Store"].nunique())

    st.divider()

    # -------------------------
    # Daily Sales Trend
    # -------------------------

    st.subheader("📈 Daily Sales Trend")

    daily_sales = df.groupby("Date")["Sales"].sum().reset_index()

    fig = px.line(daily_sales, x="Date", y="Sales", title="Daily Sales Trend")

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Monthly Sales
    # -------------------------

    st.subheader("📅 Monthly Sales")

    df["Date"] = pd.to_datetime(df["Date"])
    df["Month_Num"] = df["Date"].dt.month
    df["Month"] = df["Date"].dt.strftime("%b")

    monthly_sales = (
        df.groupby(["Month_Num", "Month"], as_index=False)["Sales"]
        .mean()
        .sort_values("Month_Num"))
        
    fig = px.bar(
        monthly_sales,
        x="Month",
        y="Sales",
        color="Sales", 
        color_continuous_scale="blues",
        text_auto=True,
        title="Average Monthly Sales",
    )
    fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Sales by Day of Week
    # -------------------------

    st.subheader("📆 Sales by Day of Week")

    dow = df.groupby("DayOfWeek")["Sales"].mean().reset_index()

    fig = px.bar(
        dow,
        x="DayOfWeek",
        y="Sales",
        color="Sales",
        color_continuous_scale="blues",
        text_auto=True,
        title="Average Sales by Day",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Promo Effect
    # -------------------------

    st.subheader("🎁 Sales with and without Promotion")

    promo = df.groupby("Promo")["Sales"].mean().reset_index()

    fig = px.pie(
        promo,
        names="Promo",
        values="Sales",
        hole=0.5,
        color="Promo",
        title="Average Sales by Promotion",
    )
    fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Top 10 Stores
    # -------------------------

    st.subheader("🏆 Top 10 Performing Stores")

    top_store = (
        df.groupby("Store")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(top_store, x="Store", y="Sales",
                 color="Sales",
        color_continuous_scale="blues",text_auto=True, title="Top 10 Stores")

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Correlation Heatmap
    # -------------------------

    st.subheader("🔥 Feature Correlation")

    corr = df.corr(numeric_only=True)

     numeric_df = df.select_dtypes(include=np.number).copy()

    # Remove constant columns
    # Constant columns cause NaN correlations
    numeric_df = numeric_df.loc[:, numeric_df.nunique(dropna=True) > 1]

    # Calculate correlation
    corr = numeric_df.corr()

    # Remove infinity values
    corr = corr.replace([np.inf, -np.inf], np.nan)

    # Replace remaining NaN values
    corr = corr.fillna(0)

    # Create heatmap
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="tealrose",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Correlation Heatmap",
    )

    fig.update_traces(textfont_size=9)

    fig.update_layout(height=700, margin=dict(l=20, r=20, t=50, b=50))

    st.plotly_chart(fig, use_container_width=True)

    with tab4:

        st.header("ℹ️ About This Project")

        st.markdown("""
            ## 📈 Sales Forecasting and Time Series Analysis

            This application predicts future store sales using a Machine Learning model
            trained on the Rossmann Store Sales dataset.

            The goal of this project is to help businesses estimate future sales,
            analyze historical trends, and support better business decisions through
            data-driven forecasting.
            """)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("🎯 Project Objective")

            st.write("""
                - Predict future sales
                - Analyze sales trends
                - Identify important business factors
                - Improve forecasting accuracy
                - Support business planning
                """)

        with col2:

            st.subheader("🛠 Technologies Used")

            st.write("""
                - Python
                - Pandas
                - NumPy
                - Scikit-learn
                - Plotly
                - Streamlit
                - Joblib
                """)

        st.divider()

        st.subheader("⚙️ Project Workflow")

        st.markdown("""
            ✔ Data Collection

            ✔ Data Cleaning

            ✔ Exploratory Data Analysis (EDA)

            ✔ Feature Engineering

            ✔ Model Training

            ✔ Model Evaluation

            ✔ Sales Prediction

            ✔ Streamlit Dashboard
            """)

        st.divider()

        st.subheader("🤖 Machine Learning Model")

        st.info("""
            Random Forest Regressor

            The model was trained using engineered features such as:

            • Store

            • Day Of Week

            • Customers

            • Promo

            • School Holiday

            • State Holiday

            • Month

            • Year

            • Week

            • Quarter

            • Lag Features

            • Rolling Mean Features
            
            • Expanding Mean Features

            """)

        st.divider()

        st.subheader("✨ Key Features")

        st.success("""
            ✅ Predict Future Store Sales

            ✅ Interactive Dashboard

            ✅ Business Analytics

            ✅ Historical Sales Trends

            ✅ Machine Learning Predictions

            ✅ Easy-to-use Interface
            """)

        st.divider()

        st.subheader("📂 Dataset")

        st.write("""
            **Dataset:** Rossmann Store Sales (Kaggle)

            The dataset contains historical information about stores,
            promotions, holidays, customer counts, and daily sales,
            making it suitable for building sales forecasting models.
            """)

        st.divider()

        st.markdown("""
                ### 👩‍💻 Developed By

                **Shubhangi Singh**

                **B.Tech Computer Science Engineering**

                **Aspiring Data Scientist**

                **Skills:** Python • SQL • Machine Learning • Streamlit • Scikit-learn
                """)
