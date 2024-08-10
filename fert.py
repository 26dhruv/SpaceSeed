from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the dataset
PATH = "Crop_recommendation.csv"
df = pd.read_csv(PATH)
crop_fertilizer = pd.read_csv("fertilizer.csv")
# Load the pre-trained RandomForest model
with open("RandomForest.pkl", "rb") as RF_Model_pkl:
    RF_model = pickle.load(RF_Model_pkl)


# Load the crop list from an Excel file
def load_crops(crops_df):
    """Load the crop list from a df."""
    return crops_df["Crop"].drop_duplicates().tolist()


crops = load_crops(crop_fertilizer)


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
        hybrid_traits=f"Traits: {hybrid_crop_df.to_html()}",
    )


# Fertilizer recommendation route
@app.route("/fertilizer", methods=["GET", "POST"])
def fertilizer():
    recommendation = ""
    if request.method == "POST":
        soil_ph = float(request.form["soil_ph"])
        nitrogen = float(request.form["nitrogen"])
        phosphate = float(request.form["phosphate"])
        potassium = float(request.form["potassium"])
        crop = request.form["crop"]

        # Generate fertilizer recommendation
        recommendation = recommend_fertilizer(
            soil_ph, nitrogen, phosphate, potassium, crop, crop_fertilizer
        )

    return render_template(
        "fertilizer.html", crops=crops, recommendation=recommendation
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


# Function to recommend fertilizer
def recommend_fertilizer(soil_ph, nitrogen, phosphate, potassium, crop, cf_df):
    result = cf_df.loc[cf_df["Crop"] == crop]
    if result.empty:
        return "Crop not found."

    n = result["N"].iloc[0]
    p = result["P"].iloc[0]
    k = result["K"].iloc[0]
    ph = result["pH"].iloc[0]

    recommendations = []

    if nitrogen < n:
        recommendations.append(
            """The N value of your soil is low.
        <br/> Please consider the following suggestions:
        <br/><br/> 1. <i>Add sawdust or fine woodchips to your soil</i> – the carbon in the sawdust/woodchips love nitrogen and will help absorb and soak up and excess nitrogen.

        <br/>2. <i>Plant heavy nitrogen feeding plants</i> – tomatoes, corn, broccoli, cabbage and spinach are examples of plants that thrive off nitrogen and will suck the nitrogen dry.

        <br/>3. <i>Water</i> – soaking your soil with water will help leach the nitrogen deeper into your soil, effectively leaving less for your plants to use.

        <br/>4. <i>Sugar</i> – In limited studies, it was shown that adding sugar to your soil can help potentially reduce the amount of nitrogen is your soil. Sugar is partially composed of carbon, an element which attracts and soaks up the nitrogen in the soil. This is similar concept to adding sawdust/woodchips which are high in carbon content.

        <br/>5. Add composted manure to the soil.

        <br/>6. Plant Nitrogen fixing plants like peas or beans.

        <br/>7. <i>Use NPK fertilizers with high N value.

        <br/>8. <i>Do nothing</i> – It may seem counter-intuitive, but if you already have plants that are producing lots of foliage, it may be best to let them continue to absorb all the nitrogen to amend the soil for your next crops.""",
        )
    elif nitrogen > n:
        recommendations.append(
            """The N value of soil is high and might give rise to weeds.
        <br/> Please consider the following suggestions:

        <br/><br/> 1. <i> Manure </i> – adding manure is one of the simplest ways to amend your soil with nitrogen. Be careful as there are various types of manures with varying degrees of nitrogen.

        <br/> 2. <i>Coffee grinds </i> – use your morning addiction to feed your gardening habit! Coffee grinds are considered a green compost material which is rich in nitrogen. Once the grounds break down, your soil will be fed with delicious, delicious nitrogen. An added benefit to including coffee grounds to your soil is while it will compost, it will also help provide increased drainage to your soil.

        <br/>3. <i>Plant nitrogen fixing plants</i> – planting vegetables that are in Fabaceae family like peas, beans and soybeans have the ability to increase nitrogen in your soil

        <br/>4. Plant ‘green manure’ crops like cabbage, corn and brocolli

        <br/>5. <i>Use mulch (wet grass) while growing crops</i> - Mulch can also include sawdust and scrap soft woods"""
        )

    if phosphate < p:
        recommendations.append(
            """The P value of your soil is low.
        <br/> Please consider the following suggestions:

        <br/><br/>1. <i>Bone meal</i> – a fast acting source that is made from ground animal bones which is rich in phosphorous.

        <br/>2. <i>Rock phosphate</i> – a slower acting source where the soil needs to convert the rock phosphate into phosphorous that the plants can use.

        <br/>3. <i>Phosphorus Fertilizers</i> – applying a fertilizer with a high phosphorous content in the NPK ratio (example: 10-20-10, 20 being phosphorous percentage).

        <br/>4. <i>Organic compost</i> – adding quality organic compost to your soil will help increase phosphorous content.

        <br/>5. <i>Manure</i> – as with compost, manure can be an excellent source of phosphorous for your plants.

        <br/>6. <i>Clay soil</i> – introducing clay particles into your soil can help retain & fix phosphorus deficiencies.

        <br/>7. <i>Ensure proper soil pH</i> – having a pH in the 6.0 to 7.0 range has been scientifically proven to have the optimal phosphorus uptake in plants.

        <br/>8. If soil pH is low, add lime or potassium carbonate to the soil as fertilizers. Pure calcium carbonate is very effective in increasing the pH value of the soil.

        <br/>9. If pH is high, addition of appreciable amount of organic matter will help acidify the soil. Application of acidifying fertilize"""
        )
    elif phosphate > p:
        recommendations.append(
            """The P value of your soil is high.
        <br/> Please consider the following suggestions:

        <br/><br/>1. <i>Avoid adding manure</i> – manure contains many key nutrients for your soil but typically including high levels of phosphorous. Limiting the addition of manure will help reduce phosphorus being added.

        <br/>2. <i>Use only phosphorus-free fertilizer</i> – if you can limit the amount of phosphorous added to your soil, you can let the plants use the existing phosphorus while still providing other key nutrients such as Nitrogen and Potassium. Find a fertilizer with numbers such as 10-0-10, where the zero represents no phosphorous.

        <br/>3. <i>Water your soil</i> – soaking your soil liberally will aid in driving phosphorous out of the soil. This is recommended as a last ditch effort.

        <br/>4. Plant nitrogen fixing vegetables to increase nitrogen without increasing phosphorous (like beans and peas).

        <br/>5. Use crop rotations to decrease high phosphorous levels"""
        )

    if potassium < k:
        recommendations.append(
            """The K value of your soil is low.
        <br/>Please consider the following suggestions:

        <br/><br/>1. Mix in muricate of potash or sulphate of potash
        <br/>2. Try kelp meal or seaweed
        <br/>3. Try Sul-Po-Mag
        <br/>4. Bury banana peels an inch below the soils surface
        <br/>5. Use Potash fertilizers since they contain high values potassium
        """
        )
    elif potassium > k:
        recommendations.append(
            """The K value of your soil is high</b>.
        <br/> Please consider the following suggestions:

        <br/><br/>1. <i>Loosen the soil</i> deeply with a shovel, and water thoroughly to dissolve water-soluble potassium. Allow the soil to fully dry, and repeat digging and watering the soil two or three more times.

        <br/>2. <i>Sift through the soil</i>, and remove as many rocks as possible, using a soil sifter. Minerals occurring in rocks such as mica and feldspar slowly release potassium into the soil slowly through weathering.

        <br/>3. Stop applying potassium-rich commercial fertilizer. Apply only commercial fertilizer that has a '0' in the final number field. Commercial fertilizers use a three number system for measuring levels of nitrogen, phosphorous and potassium. The last number stands for potassium. Another option is to stop using commercial fertilizers all together and to begin using only organic matter to enrich the soil.

        <br/>4. Mix crushed eggshells, crushed seashells, wood ash or soft rock phosphate to the soil to add calcium. Mix in up to 10 percent of organic compost to help amend and balance the soil.

        <br/>5. Use NPK fertilizers with low K levels and organic fertilizers since they have low NPK values.

        <br/>6. Grow a cover crop of legumes that will fix nitrogen in the soil. This practice will meet the soil’s needs for nitrogen without increasing phosphorus or potassium.
        """
        )

    if not recommendations:
        recommendations.append("Soil conditions are perfect for the selected crop.")

    return " ".join(recommendations)


if __name__ == "__main__":
    app.run(debug=True)
