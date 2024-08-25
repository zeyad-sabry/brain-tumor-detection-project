import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

def predict_image(model, image_path, target_size=(224, 224)):
    
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    return prediction[0][0]

def main():
    st.title("Brain MRI Tumor Detector")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "JPG"])
    
    path = r'C:\Users\zeyad\OneDrive\Desktop\CVProject\model.keras'        
    model = tf.keras.models.load_model(path)
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        image = Image.open(uploaded_file)
        prediction = predict_image(model, uploaded_file)
        if prediction > 0.5:
            st.write("Prediction: **Positive** (Class 1) with accuracy: " , prediction)
        else:
            st.write("Prediction: **Negative** (Class 0) with accuracy: " , prediction)

if __name__ == '__main__':
    main()