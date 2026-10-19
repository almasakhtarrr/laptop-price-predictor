
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "laptop_price_model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv(BASE_DIR / "laptop_data.csv")
    df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)

    df["Ram"] = pd.to_numeric(
        df["Ram"].astype(str).str.replace("GB", "", regex=False),
        errors="raise"
    )

    df["Weight"] = pd.to_numeric(
        df["Weight"].astype(str).str.replace("kg", "", regex=False),
        errors="raise"
    )

    return df

model = load_model()
df = load_data()

st.title("💻 Laptop Price Predictor")
st.write(
    "Enter your laptop specifications to estimate its price."
)

st.divider()

with st.form("laptop_form"):
    st.subheader("Laptop specifications")

    col1, col2 = st.columns(2)

    with col1:
        company = st.selectbox(
            "Brand",
            sorted(df["Company"].dropna().unique())
        )

        laptop_type = st.selectbox(
            "Laptop type",
            sorted(df["TypeName"].dropna().unique())
        )

        ram = st.selectbox(
            "RAM (GB)",
            sorted(df["Ram"].dropna().unique())
        )

        inches = st.number_input(
            "Screen size (inches)",
            min_value=10.0,
            max_value=20.0,
            value=15.6,
            step=0.1
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=0.5,
            max_value=5.0,
            value=1.8,
            step=0.1
        )

    with col2:
        resolution = st.selectbox(
            "Screen resolution",
            sorted(df["ScreenResolution"].dropna().unique())
        )

        cpu = st.selectbox(
            "Processor",
            sorted(df["Cpu"].dropna().unique())
        )

        memory = st.selectbox(
            "Storage",
            sorted(df["Memory"].dropna().unique())
        )

        gpu = st.selectbox(
            "Graphics card",
            sorted(df["Gpu"].dropna().unique())
        )

        os_name = st.selectbox(
            "Operating system",
            sorted(df["OpSys"].dropna().unique())
        )

    submitted = st.form_submit_button(
        "Predict Price",
        use_container_width=True
    )

if submitted:
    input_data = pd.DataFrame([{
        "Company": company,
        "TypeName": laptop_type,
        "Inches": float(inches),
        "ScreenResolution": resolution,
        "Cpu": cpu,
        "Ram": float(ram),
        "Memory": memory,
        "Gpu": gpu,
        "OpSys": os_name,
        "Weight": float(weight)
    }])

    try:
        prediction = float(model.predict(input_data)[0])

        st.divider()
        st.subheader("Estimated laptop price")

        st.metric(
            label="Predicted price",
            value=f"₹{prediction:,.0f}"
        )

        st.caption(
            "This is a model-based estimate, not a guaranteed market price."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Check that your saved model was trained with the same "
            "feature names and preprocessing steps."
        )
