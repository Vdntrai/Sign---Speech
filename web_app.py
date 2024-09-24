import cv2
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO

st.set_page_config(page_title='ASL Recognition')
st.title('Sign Language Recognition')
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_best_model():
    try:
        best_model = keras.models.load_model('models/experiment-dropout-0.keras')
    except ValueError:
        best_model = keras.models.load_model('models/experiment-dropout-0.h5')
    return best_model

@st.cache_data
def get_label_binarizer():
    train_df = pd.read_csv('data/alphabet/sign_mnist_train.csv')
    y = train_df['label']
    label_binarizer = LabelBinarizer()
    y = label_binarizer.fit_transform(y)
    return label_binarizer

def preprocess_image(image):
    image = tf.reshape(image, [image.shape[0], image.shape[1], 1])
    image = image / 255
    image = tf.image.resize(image, [28, 28], preserve_aspect_ratio=True)
    preprocessed_image = np.ones((1, 28, 28, 1))
    preprocessed_image[0, :image.shape[0], :image.shape[1], :] = image
    return preprocessed_image

def predict_letter(image, model, binarizer):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    index_to_letter_map = {i: chr(ord('a') + i) for i in range(26)}
    letter = index_to_letter_map[binarizer.inverse_transform(prediction)[0]]
    return letter

def main():
    best_model = get_best_model()
    label_binarizer = get_label_binarizer()

    st.subheader('Real-Time ASL Recognition')
    st.write('Start the webcam feed and make sure to show the ASL sign clearly.')

    frame_placeholder = st.empty()
    prediction_placeholder = st.empty()

    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

    current_sentence = ''
    buffer = ''
    last_prediction = ''
    idle_frames = 0
    buffer_threshold = 5  # Number of frames before a new character is added
    idle_threshold = 30   # Number of idle frames before resetting

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = np.array(gray, dtype='float32')

        letter = predict_letter(image, best_model, label_binarizer)

        if letter == '':  # If no clear prediction, consider it idle
            idle_frames += 1
            if idle_frames >= idle_threshold:
                buffer = ''
                last_prediction = ''
                current_sentence += ' '
        else:
            idle_frames = 0
            if letter != last_prediction:
                buffer += letter
                last_prediction = letter

            if len(buffer) >= buffer_threshold:
                if buffer.strip() == 'space':  # Handle space
                    current_sentence += ' '
                elif buffer.strip() == 'punctuation':  # Handle punctuation
                    current_sentence += '.'
                else:
                    current_sentence += buffer.strip()

                buffer = ''
                last_prediction = ''

        # Display video frame and current sentence
        frame_placeholder.image(frame, channels='BGR', use_column_width=True)
        prediction_placeholder.write(f'Current Sentence: {current_sentence}')

    cap.release()

if __name__ == "__main__":
    main()
