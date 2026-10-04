
import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Fish Disease Detection AI",
    page_icon="🐟",
    layout="centered"
)

# ==========================================
# DISEASE CLASSES
# ==========================================

class_names = [
    "Bacterial Red disease",
    "Bacterial diseases - Aeromoniasis",
    "Bacterial gill disease",
    "EUS Disease",
    "Fungal diseases Saprolegniasis",
    "Healthy Fish",
    "Parasitic diseases",
    "Viral diseases White tail disease"
]

# ==========================================
# LOAD MODELS
# ==========================================

@st.cache_resource
def load_models():

    svm_model = joblib.load(
        "FINAL_hybrid_svm_86_25.pkl"
    )

    scaler = joblib.load(
        "FINAL_scaler_no_leakage.pkl"
    )

    interpreter = tf.lite.Interpreter(
        model_path="resnet50_feature_extractor.tflite"
    )

    interpreter.allocate_tensors()

    return svm_model, scaler, interpreter


svm_model, scaler, interpreter = load_models()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==========================================
# WEBSITE TITLE
# ==========================================

st.title("🐟 Fish Disease Detection AI")

st.write(
    "Upload a fish image and the AI system will "
    "analyze it and predict the possible disease."
)

st.divider()

# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a fish image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Fish Image",
        use_container_width=True
    )

    if st.button("🔍 Analyze Fish"):

        # Resize image
        img = image.resize((224, 224))

        # Convert to NumPy
        img_array = np.array(img).astype(np.float32)

        # --------------------------------------
        # RGB FEATURES
        # --------------------------------------

        rgb_features = img_array.mean(axis=(0, 1))

        # --------------------------------------
        # RESNET50 INPUT
        # --------------------------------------

        # ResNet50 preprocessing
        x = np.expand_dims(img_array, axis=0)

        from tensorflow.keras.applications.resnet50 import preprocess_input

        x = preprocess_input(x)

        # --------------------------------------
        # TFLITE FEATURE EXTRACTION
        # --------------------------------------

        interpreter.set_tensor(
            input_details[0]["index"],
            x
        )

        interpreter.invoke()

        deep_features = interpreter.get_tensor(
            output_details[0]["index"]
        )

        # --------------------------------------
        # HYBRID FEATURES
        # --------------------------------------

        hybrid_features = np.concatenate(
            [
                deep_features,
                rgb_features.reshape(1, -1)
            ],
            axis=1
        )

        # --------------------------------------
        # SCALE FEATURES
        # --------------------------------------

        scaled_features = scaler.transform(
            hybrid_features
        )

        # --------------------------------------
        # PREDICTION
        # --------------------------------------

        prediction = svm_model.predict(
            scaled_features
        )

        predicted_index = int(prediction[0])

        predicted_disease = class_names[
            predicted_index
        ]

        probabilities = svm_model.predict_proba(
            scaled_features
        )

        confidence = (
            float(np.max(probabilities[0])) * 100
        )

        # ======================================
        # DISPLAY RESULT
        # ======================================

        st.divider()

        st.subheader("🐟 Prediction Result")

        st.success(
            f"Predicted Disease: {predicted_disease}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )
