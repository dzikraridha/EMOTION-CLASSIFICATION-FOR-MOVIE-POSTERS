import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.metrics import Metric

# Define the class labels (emotions)
emotion_labels = ['Fear', 'Disgust', 'Surprised', 'Neutral', 'Happy', 'Sad', 'Angry']

@register_keras_serializable()
def multi_label_accuracy(y_true, y_pred):
    # Custom metric to calculate multi-label accuracy
    y_true = tf.cast(y_true, tf.float32)  # Cast y_true to float32
    y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Convert predictions to binary
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))

def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.round(y_pred), tf.float32)

    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return tf.reduce_mean(f1)

# Load model
model = load_model('movie_emotion_model_fin.keras', custom_objects={
    'multi_label_accuracy': multi_label_accuracy,
    'f1_score': f1_score
})


# Function to preprocess the uploaded image
def load_and_preprocess_image(uploaded_image):
    img = image.load_img(uploaded_image, target_size=(224, 224))  # Adjust size as per your model
    img_array = image.img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize (if needed)
    return img_array

# Function to predict emotions and display results
def predict_emotions(model, uploaded_image, threshold=0.5):
    # Preprocess the image
    img_array = load_and_preprocess_image(uploaded_image)
    
    # Get predictions
    predictions = model.predict(img_array)[0]  # Extract raw predictions
    
    # Filter predictions exceeding the threshold
    filtered_predictions = [(emotion, prob) for emotion, prob in zip(emotion_labels, predictions) if prob > threshold]
    
    # Sort by probabilities in descending order
    filtered_predictions = sorted(filtered_predictions, key=lambda x: x[1], reverse=True)
    
    # Extract the top 1–3 predicted emotions
    top_predictions = [emotion for emotion, _ in filtered_predictions[:3]]
    
    # Create a DataFrame for raw probabilities
    prediction_df = pd.DataFrame({'Emotion': emotion_labels, 'Probability': predictions})
    
    return top_predictions, prediction_df

# Streamlit app UI
st.title("Emotion Classification for Movie Posters")
st.write("Upload a movie poster to predict the emotions:")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_container_width=True)
    st.write("")
    
    # Predict emotions
    top_emotions, prediction_df = predict_emotions(model, uploaded_file)
    
    # Display top predicted emotions
    if top_emotions:
        st.write("Top Predicted Emotions: ", ", ".join(top_emotions))
    else:
        st.write("No emotions detected above the threshold.")
    
    # Display raw predictions
    st.write("Raw Predictions:")
    st.dataframe(prediction_df)
