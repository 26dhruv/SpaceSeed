from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the dataset
PATH = "Crop_recommendation.csv"
df = pd.read_csv(PATH)

# Load the pre-trained RandomForest model
with open("RandomForest.pkl", "rb") as RF_Model_pkl:
    RF_model = pickle.load(RF_Model_pkl)


# Home route
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    # Get input values from the form
    int_features = [
        float(x) for x in request.form.values() if x != "Earth" and x != "Mars"
    ]
    data = [np.array(int_features)]
    planet = request.form["planet"]

    # Adjust features based on the planet selected
    if planet == "Mars":
        data = adjust_for_mars(data)

    # Predict the top two crops
    proba = RF_model.predict_proba(data)
    top_two_indices = np.argsort(proba[0])[-2:][::-1]
    top_two_crops = RF_model.classes_[top_two_indices]

    # Get crop traits for hybridization
    crop1_traits = df[df["label"] == top_two_crops[0]].iloc[0][
        ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    ]
    crop2_traits = df[df["label"] == top_two_crops[1]].iloc[0][
        ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    ]

    crop1_df = pd.DataFrame([crop1_traits])
    crop2_df = pd.DataFrame([crop2_traits])

    # Hybridize crops
    hybrid_crop_df = hybridize_crops(crop1_df, crop2_df, method="average")

    # Predict hybrid crop's performance
    hybrid_prediction = RF_model.predict(hybrid_crop_df)[0]

    return render_template(
        "index.html",
        prediction_text=f"Top 2 Crops: {top_two_crops[0]}, {top_two_crops[1]}",
        hybrid_text=f"Predicted Hybrid Crop: {hybrid_prediction} Dominant",
        hybrid_traits=f"Traits: {hybrid_crop_df}",
    )


# Function to adjust data for Mars
def adjust_for_mars(data):
    # Simple example of adjusting temperature and rainfall for Mars
    data[0][3] = data[0][3] - 30  # Adjust temperature
    data[0][6] = data[0][6] * 0.1  # Adjust rainfall
    return data


# Hybridization function
def hybridize_crops(crop1, crop2, method="average"):
    hybrid = {}
    if method == "average":
        for trait in crop1.columns:
            hybrid[trait] = (crop1[trait].values[0] + crop2[trait].values[0]) / 2
    elif method == "random":
        for trait in crop1.columns:
            hybrid[trait] = np.random.choice(
                [crop1[trait].values[0], crop2[trait].values[0]]
            )
    elif method == "weighted":
        weight1 = 0.6
        weight2 = 0.4
        for trait in crop1.columns:
            hybrid[trait] = (
                weight1 * crop1[trait].values[0] + weight2 * crop2[trait].values[0]
            )
    return pd.DataFrame([hybrid])


if __name__ == "__main__":
    app.run(debug=True)
