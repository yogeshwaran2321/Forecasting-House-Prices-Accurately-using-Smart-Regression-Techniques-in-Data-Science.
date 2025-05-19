import gradio as gr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# Load and preprocess data (simplified)
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
df = pd.read_csv(url)
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
df = df.drop('ocean_proximity', axis=1)
df['median_house_value'] = np.log1p(df['median_house_value'])

# Prepare data
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Prediction function
def predict_house_value(longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
                        population, households, median_income):
    input_data = np.array([[longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
                            population, households, median_income]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return np.expm1(prediction[0])  # Reverse log transformation

# Create Gradio interface
iface = gr.Interface(
    fn=predict_house_value,
    inputs=[
        gr.Slider(-124, -114, step=0.1, label="Longitude"),
        gr.Slider(32, 42, step=0.1, label="Latitude"),
        gr.Slider(0, 52, step=1, label="Housing Median Age"),
        gr.Slider(0, 40000, step=100, label="Total Rooms"),
        gr.Slider(0, 7000, step=10, label="Total Bedrooms"),
        gr.Slider(0, 50000, step=100, label="Population"),
        gr.Slider(0, 7000, step=10, label="Households"),
        gr.Slider(0, 15, step=0.1, label="Median Income")
    ],
    outputs="text",
    title="California House Price Predictor",
    description="Enter features to predict the median house value (in $)."
)

# Launch interface
iface.launch()