
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Rainfall Prediction",
    page_icon="🌧️",
    layout="centered"
)

# Load the trained model
# Cache the model to avoid reloading it on every rerun, improving performance.
@st.cache_resource
def load_model():
    return joblib.load("rainfall_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file 'rainfall_model.pkl' not found. "
        "Please train the model first by running:\n\n"
        "`python train_model.py`\n\n"
        "Make sure 'rainfall_model.pkl' is saved in the same directory as this app. "
        "For more information, see the project documentation or README."
    )
    st.stop()

# Website heading
st.title("🌧️ Rainfall Prediction System")
st.write(
    "Enter the weather conditions to predict "
    "whether rainfall is expected."
)

st.info(
    "Educational demo: this model uses only 15 "
    "illustrative records and is not suitable for real forecasts."
)

# Input form
with st.form("weather_form"):
    st.subheader("Enter Weather Details")

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-10.0,
        max_value=60.0,
        value=25.0
    )

    humidity = st.slider(
        "Humidity (%)",
        min_value=0,
        max_value=100,
        value=80
    )

    pressure = st.number_input(
        "Atmospheric Pressure (hPa)",
        min_value=850.0,
        max_value=1100.0,
        value=1005.0
    )

    wind_speed = st.number_input(
        "Wind Speed (km/h)",
        min_value=0.0,
        max_value=300.0,
        value=15.0
    )

    cloud_cover = st.slider(
        "Cloud Cover (%)",
        min_value=0,
        max_value=100,
        value=80
    )

    submitted = st.form_submit_button(
        "Predict Rainfall",
        use_container_width=True
    )

# Make prediction
if submitted:
    input_data = pd.DataFrame([{
        "Temperature": temperature,
        "Humidity": humidity,
        "Pressure": pressure,
        "WindSpeed": wind_speed,
        "CloudCover": cloud_cover
    }])

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == "Yes":
        st.warning("🌧️ Prediction: Rain Tomorrow")
    elif prediction == "No":
        st.success("☀️ Prediction: No Rain Tomorrow")

    # Show the model's class probabilities if available
    try:
        probabilities = model.predict_proba(input_data)[0]
        probability_by_class = dict(
            zip(model.classes_, probabilities)
        )

        st.write("Model probability estimates:")
        st.write(
            f"Rain: {probability_by_class.get('Yes', 0):.1%}"
        )
        st.write(
            f"No rain: {probability_by_class.get('No', 0):.1%}"
        )

        st.caption(
            "These are model estimates, not calibrated "
            "real-world weather probabilities."
        )
    except AttributeError:
        st.info("Probability estimates are not available for this model.")
