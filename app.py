import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Crop Health Prediction",
    page_icon="🌱",
    layout="wide"
)

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""
<style>

.main{
    background:#f5fff5;
}

.big-title{
    text-align:center;
    color:#1b5e20;
    font-size:42px;
    font-weight:bold;
}

.sub-title{
    text-align:center;
    color:#2e7d32;
    font-size:20px;
}

.result-box{
    padding:15px;
    border-radius:10px;
    background:#e8f5e9;
    border:2px solid green;
}

.stButton>button{
    width:100%;
    height:55px;
    font-size:18px;
    font-weight:bold;
    border-radius:10px;
}

.footer{
    text-align:center;
    color:gray;
    padding:20px;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================

st.markdown(
    "<div class='big-title'>🌾 Crop Health Prediction System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>AI Powered Smart Agriculture Dashboard</div>",
    unsafe_allow_html=True
)

st.divider()

# ======================================================
# LOAD MODELS
# ======================================================

feature_model = joblib.load("model.pkl")

cnn_model = load_model("crop_disease_cnn.keras")

CLASS_NAMES = [
    "Healthy",
    "Powdery",
    "Rust"
]

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2909/2909760.png",
    width=120
)

st.sidebar.title("🌱 Navigation")

st.sidebar.markdown("""
### 📊 Dashboard

✅ Feature Prediction

✅ Image Prediction

🌾 Dataset : Crop Monitoring

🤖 AI Powered
""")

st.sidebar.link_button(
    "📄 Dataset Information",
    "https://mohit260805.github.io/Crop-Health-Prediction/Dataset_info.html"
)

st.sidebar.success("Models Loaded Successfully")

# ======================================================
# TABS
# ======================================================

tab1, tab2 = st.tabs(
    [
        "📊 Feature Prediction",
        "📷 Image Prediction"
    ]
)

# ======================================================
# IMAGE PREDICTION TAB
# ======================================================

with tab2:

    st.header("📷 Crop Disease Detection")

    st.write(
        "Upload a crop leaf image or capture one using your camera."
    )

    uploaded_file = st.file_uploader(
        "Upload Leaf Image",
        type=["jpg", "jpeg", "png"]
    )

    camera_file = st.camera_input(
        "Capture Image"
    )

    image_file = uploaded_file if uploaded_file is not None else camera_file

    if image_file is not None:

        image = Image.open(image_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button(
            "🔍 Predict Disease from Image",
            key="image_predict"
        ):

            img = image.resize((224,224))

            img_array = np.array(img)

            img_array = img_array.astype("float32") / 255.0

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            prediction = cnn_model.predict(img_array)

            predicted_index = np.argmax(prediction)

            predicted_class = CLASS_NAMES[predicted_index]

            confidence = np.max(prediction) * 100

            st.success(
                f"Prediction : {predicted_class}"
            )

            st.info(
                f"Confidence : {confidence:.2f}%"
            )

            if predicted_class == "Healthy":
                st.balloons()

            elif predicted_class == "Rust":
                st.error(
                    "⚠ Rust disease detected. Consider fungicide treatment."
                )

            elif predicted_class == "Powdery":
                st.warning(
                    "⚠ Powdery disease detected. Monitor the crop carefully."
                )

# ======================================================
# FEATURE PREDICTION TAB
# ======================================================

with tab1:
        # ======================================================
    # FEATURE INPUT SECTION
    # ======================================================

    st.subheader("📊 Enter Crop Features")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📷 Image Features")

        High_Resolution_RGB = 1 if st.selectbox(
            "High Resolution RGB",
            ["Yes", "No"]
        ) == "Yes" else 0

        Multispectral_Images = 1 if st.selectbox(
            "Multispectral Images",
            ["Yes", "No"]
        ) == "Yes" else 0

        Thermal_Images = 1 if st.selectbox(
            "Thermal Images",
            ["Yes", "No"]
        ) == "Yes" else 0

        Temporal_Images = 1 if st.selectbox(
            "Temporal Images",
            ["Yes", "No"]
        ) == "Yes" else 0

        Spatial_Resolution = st.number_input(
            "Spatial Resolution",
            value=1.2
        )

        GPS_Coordinates = st.number_input(
            "GPS Coordinates",
            min_value=100000,
            max_value=999999,
            value=550000
        )

        Field_Boundaries = st.selectbox(
            "Field Boundaries",
            ["Type 1", "Type 2", "Type 3"]
        )

        Field_Boundaries = {
            "Type 1": 1,
            "Type 2": 2,
            "Type 3": 3
        }[Field_Boundaries]

        Elevation_Data = st.number_input(
            "Elevation Data",
            value=100.0
        )

        Canopy_Coverage = st.number_input(
            "Canopy Coverage",
            value=35.0
        )

        NDVI = st.number_input(
            "NDVI",
            min_value=-1.5,
            max_value=1.5,
            value=0.5
        )

        SAVI = st.number_input(
            "SAVI",
            min_value=-1.5,
            max_value=1.5,
            value=0.5
        )

    with col2:

        st.subheader("🌿 Crop & Environment")

        Chlorophyll_Content = st.number_input(
            "Chlorophyll Content",
            min_value=0.0,
            max_value=8.0,
            value=1.0
        )

        Leaf_Area_Index = st.number_input(
            "Leaf Area Index",
            min_value=0.0,
            max_value=6.0,
            value=1.5
        )

        Crop_Stress_Indicator = st.number_input(
            "Crop Stress Indicator",
            min_value=0,
            max_value=100,
            value=50
        )

        Temperature = st.number_input(
            "Temperature (°C)",
            value=25.0
        )

        Humidity = st.number_input(
            "Humidity (%)",
            value=60.0
        )

        Rainfall = st.number_input(
            "Rainfall (mm)",
            value=20.0
        )

        Wind_Speed = st.number_input(
            "Wind Speed",
            value=2.0
        )

        Soil_Moisture = st.number_input(
            "Soil Moisture",
            value=20.0
        )

        Soil_pH = st.number_input(
            "Soil pH",
            value=6.5
        )

        Organic_Matter = st.number_input(
            "Organic Matter",
            value=2.0
        )

    st.divider()

    st.subheader("🚜 Additional Features")

    c1, c2, c3 = st.columns(3)

    with c1:

        Pest_Hotspots = 1 if st.selectbox(
            "Pest Hotspots",
            ["High", "Low"]
        ) == "High" else 0

        Weed_Coverage = st.number_input(
            "Weed Coverage",
            value=2.5
        )

    with c2:

        Pest_Damage = st.number_input(
            "Pest Damage",
            value=50
        )

        Crop_Growth_Stage = st.selectbox(
            "Crop Growth Stage",
            [1, 2, 3, 4]
        )

    with c3:

        Crop_Type = st.radio(
            "Crop Type",
            ["Wheat", "Maize", "Rice"]
        )

        Crop_Type = {
            "Wheat": 1,
            "Maize": 2,
            "Rice": 3
        }[Crop_Type]

    Bounding_Boxes = st.number_input(
        "Bounding Boxes",
        min_value=0,
        max_value=9,
        value=1
    )

    Water_Flow = st.number_input(
        "Water Flow",
        value=25.0
    )

    Drainage_Features = 1 if st.selectbox(
        "Drainage Features",
        ["Poor", "Good"]
    ) == "Good" else 0

        # ======================================================
    # FEATURE PREDICTION
    # ======================================================

    st.divider()

    if st.button("🔍 Predict Crop Health", use_container_width=True):

        data = pd.DataFrame([[
            High_Resolution_RGB,
            Multispectral_Images,
            Thermal_Images,
            Temporal_Images,
            Spatial_Resolution,
            GPS_Coordinates,
            Field_Boundaries,
            Elevation_Data,
            Canopy_Coverage,
            NDVI,
            SAVI,
            Chlorophyll_Content,
            Leaf_Area_Index,
            Crop_Stress_Indicator,
            Temperature,
            Humidity,
            Rainfall,
            Wind_Speed,
            Soil_Moisture,
            Soil_pH,
            Organic_Matter,
            Pest_Hotspots,
            Weed_Coverage,
            Pest_Damage,
            Crop_Growth_Stage,
            Crop_Type,
            Bounding_Boxes,
            Water_Flow,
            Drainage_Features
        ]])

        with st.spinner("🔍 Analyzing crop data..."):

            prediction = feature_model.predict(data)

            confidence = None

            try:
                probability = feature_model.predict_proba(data)
                confidence = float(np.max(probability)) * 100
            except Exception:
                pass

        st.divider()

        st.subheader("📋 Prediction Result")

        if prediction[0] == 0:

            st.success("✅ Crop Health : Healthy")

            st.balloons()

        else:

            st.error("⚠ Crop Health : Disease Detected")

        if confidence is not None:

            st.progress(confidence / 100)

            st.info(f"Confidence : {confidence:.2f}%")

        st.subheader("📈 Input Summary")

        summary = pd.DataFrame({
            "Feature": [
                "Temperature",
                "Humidity",
                "Rainfall",
                "Soil Moisture",
                "Soil pH",
                "NDVI",
                "SAVI",
                "Crop Type"
            ],
            "Value": [
                Temperature,
                Humidity,
                Rainfall,
                Soil_Moisture,
                Soil_pH,
                NDVI,
                SAVI,
                Crop_Type
            ]
        })

        st.dataframe(summary, use_container_width=True)

# ======================================================
# FOOTER
# ======================================================

st.divider()

st.markdown(
    """
<div class="footer">

🌱 <b>Smart Agriculture Dashboard</b><br>
Developed by <b>Mohit Gaur</b><br>
📧 mohitgaurpbc@gmail.com

</div>
""",
    unsafe_allow_html=True
)

