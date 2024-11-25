import streamlit as st
import pandas as pd
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from PIL import Image
import joblib

@st.cache(allow_output_mutation=True)
def train_model():
    model = LogisticRegression(tol=0.1, max_iter=1000)
    
    df = pd.read_csv('train.csv')
    target = df["label"]
    features = df.drop(columns="label", axis=1)
    model.fit(features, target)
    return model

def main():
    st.title('Digit Predictor')

    model = train_model()

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=200)

        img_array = np.array(image)
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        width, height = 28, 28
        img_resized = cv2.resize(img_gray, (width, height))
        _, binary_image = cv2.threshold(img_resized, 180, 255, cv2.THRESH_BINARY)
        
        binary1 = binary_image.reshape(-1)

        prediction = model.predict([binary1])
        st.write("#### Predicted number from the image:", prediction[0])

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Unable to open webcam.")
        return

    st.write("Click the button below to capture an image from the webcam.")

    if st.button("Capture"):
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            st.image(pil_image, caption='Captured Image', width=200)

            img_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

            width, height = 28, 28
            img_resized = cv2.resize(img_gray, (width, height))
            _, binary_image = cv2.threshold(img_resized, 180, 255, cv2.THRESH_BINARY)
            binary_array = (binary_image / 255).astype(int)
            binary1 = binary_image.reshape(-1)

            prediction = model.predict([binary1])
            st.write("#### Predicted number from the captured image:", prediction[0])

    cap.release()

if __name__ == '__main__':
    main()
