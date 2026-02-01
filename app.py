import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("linear_reg_model.pkl", "rb") as file:
    model = pickle.load(file)

# App title
st.set_page_config(page_title="ML Prediction App", layout="centered")
st.title("📊 Machine Learning Prediction App")
st.write("Predict output using a trained Linear Regression model")

# Sidebar
st.sidebar.header("🔧 App Controls")
st.sidebar.info("Enter values and click Predict")

# Input fields (modify count based on your model features)
feature1 = st.number_input("Enter Feature 1", min_value=0.0, step=0.1)
feature2 = st.number_input("Enter Feature 2", min_value=0.0, step=0.1)
feature3 = st.number_input("Enter Feature 3", min_value=0.0, step=0.1)

# Prediction
if st.button("🔮 Predict"):
    input_data = np.array([[feature1, feature2, feature3]])
    prediction = model.predict(input_data)

    st.success(f"✅ Predicted Value: **{prediction[0]:.2f}**")

# Extra info
st.markdown("---")
st.markdown("### ℹ️ About the App")
st.write("""
- Built using **Streamlit**
- Uses a **pre-trained Linear Regression model**
- Simple & interactive UI
""")

# Footer
st.markdown("👨‍💻 Developed for AIML Project")
