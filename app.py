# app.py - Height Classifier Web App
import streamlit as st
from PIL import Image
import os
from predict import predict_single_image

st.set_page_config(page_title="Height Classifier", page_icon="📏", layout="centered")
st.title("📏 Height Classifier")
st.markdown("Upload a full body photo to classify height as **Short, Moderate or Tall**")

with st.sidebar:
    st.header("About")
    st.write("**Model:** EfficientNetB0_B")
    st.write("**Framework:** TensorFlow")
    st.write("**Classes:** Short · Moderate · Tall")
    st.markdown("---")
    st.caption("Run with: streamlit run app.py")

st.markdown("---")
uploaded_file = st.file_uploader("📁 Upload a photo of a person", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, width=350, caption="Uploaded Image")
    st.markdown("")
    
    if st.button("🔍 Predict Height", type="primary", use_container_width=True):
        temp_path = "temp_image.jpg"
        image.convert("RGB").save(temp_path)
        
        with st.spinner("Analyzing image..."):
            try:
                predicted_class, confidence = predict_single_image(temp_path)
                st.markdown("---")
                
                if predicted_class == "Tall": emoji = "👑"
                elif predicted_class == "Moderate": emoji = "🏆"
                else: emoji = "🏅"
                
                st.success(f"{emoji} **Result: {predicted_class.upper()}**")
                st.metric(label="Confidence", value=f"{confidence:.1f}%")
                
                if confidence >= 70: st.info("✅ High confidence prediction")
                elif confidence >= 50: st.warning("⚠️ Moderate confidence — try a clearer photo")
                else: st.error("❌ Low confidence — use a clear full body standing photo")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.info("Make sure best_model.keras exists in this folder.")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
else:
    st.info("👆 Upload a photo above to begin")
    st.markdown("")
    st.markdown("**Tips for best results:**")
    st.write("• Use a clear full body photo")
    st.write("• Person should be standing upright")
    st.write("• Good lighting helps accuracy")

st.markdown("---")
st.caption("Height Classifier | TensorFlow + Streamlit")
