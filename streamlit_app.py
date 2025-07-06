import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Streamlit page config
st.set_page_config(
    page_title="🐾 Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered"
)

# Sidebar
st.sidebar.title("🐾 About This App")
st.sidebar.write("""
Upload an image of a cat or dog,  
then click **Classify** to see what our AI predicts!  
""")
st.sidebar.markdown("""
- **Model**: MobileNetV2  
- **Framework**: TensorFlow / Keras  
- **Deployment**: Streamlit  
""")
st.sidebar.info("🚀 Made with ❤️ for animal lovers.")

# Title
st.title("🐾 Cat vs Dog Image Classifier")
st.write("Upload an image and click **Classify** to see the result.")

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/cat_dog_model.keras")
    return model

model = load_model()

# Upload
uploaded_file = st.file_uploader("Choose a JPG/PNG file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # File size check
    if uploaded_file.size > 5 * 1024 * 1024:
        st.error("❌ File too large! Please upload under 5MB.")
    else:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        
        # Add a Classify button
        if st.button("🚀 Classify"):
            with st.spinner("Classifying... please wait"):
                # Preprocess
                img_resized = img.resize((224, 224))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                # Predict
                prediction = model.predict(img_array)[0][0]
                confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
                label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"

                # Results
                st.markdown(f"### ✅ **Prediction**: {label}")
                st.success(f"Confidence: {confidence:.2f}%")
                st.progress(int(confidence))

                # confetti
                if prediction > 0.5:
                    st.balloons()
                else:
                    st.snow()
                
                st.markdown("---")
                st.markdown("##### 🔎 Notes:")
                st.caption("""
                - The model is trained on thousands of cat/dog images.  
                - Works best with clear, centered pet photos.  
                - Share the app with your friends!  
                """)
else:
    st.info("👈 Please upload an image to get started.")

