import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from joblib import load

# Tải mô hình đã lưu
model = tf.keras.models.load_model('face_AI_model.h5')

# Tải LabelEncoder đã lưu
label_encoder = load('label_encoder.pkl')

# Kích thước ảnh chuẩn hóa
IMG_SIZE = (100, 100)

# Hàm dự đoán
def predict(image):
    image = np.array(image)
    resized_image = cv2.resize(image, IMG_SIZE)  # Resize ảnh về kích thước chuẩn hóa
    normalized_image = resized_image / 255.0
    prediction = model.predict(np.expand_dims(normalized_image, axis=0))
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

# Tiêu đề của ứng dụng
st.title("Face Recognition App")

# Tải ảnh lên
uploaded_file = st.file_uploader("Chọn một ảnh để dự đoán", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh đã tải lên', use_column_width=True)
    
    # Dự đoán khi nhấn nút
    if st.button("Dự đoán"):
        prediction = predict(image)
        st.write(f"Dự đoán: {prediction}")
