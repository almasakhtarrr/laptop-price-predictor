
# Laptop Price Predictor

A machine learning web application that estimates laptop prices based on hardware specifications.

## Features

- Interactive Streamlit interface
- Brand and laptop type selection
- RAM, processor, storage and GPU inputs
- Screen size, resolution, operating system and weight inputs
- Automated preprocessing with Scikit-learn
- Price prediction using Random Forest Regression

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Joblib

## Run Locally

1. Clone this repository.
2. Create and activate a virtual environment.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Ensure `laptop_data.csv` and the trained
   `laptop_price_model.pkl` are available in the project folder.
5. Launch the application:

   `streamlit run app.py`

## How It Works

The application accepts laptop specifications, preprocesses
the features using a trained Scikit-learn pipeline, and
returns an estimated laptop price.

## Limitations

Predictions depend on the training dataset and model performance.
The estimate may differ from actual market prices.

## Author

Almas Akhtar Shah
