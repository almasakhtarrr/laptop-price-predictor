import pandas as pd

from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("C:\\Users\\almas\\OneDrive\\Desktop\\laptop price predictor\\laptop_data.csv", encoding='latin-1')

df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)

df["Ram"] = df["Ram"].str.replace("GB", "", regex=False).astype(float)
df["Weight"]=df["Weight"].str.replace("kg", "", regex=False).astype(float)

X = df.drop(columns=["Price"])
y = df["Price"]

numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("numerical features: ", numerical_features)
print("categorical features: ", categorical_features)

#6 build pipelines
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")), 
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])  

#7 split the data before gitting preprocessing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#8 create a pipeline with preprocessor and model (ml pipeline)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

#9 train the model
model.fit(X_train, y_train)

#streamlit code to display the model training status and evaluation metrics
import joblib

joblib.dump(model, "laptop_price_model.pkl")
print("Model saved successfully!")

#10 evaluate the model
print("\nModel trained successfully!")

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("R² score:", round(model.score(X_test, y_test), 4))

# 11. Display sample predictions
predictions = model.predict(X_test.iloc[:5])

print("\nSample predictions:")
for actual, predicted in zip(y_test.iloc[:5], predictions):
    print(f"Actual: {actual:.2f} | Predicted: {predicted:.2f}")

