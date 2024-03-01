# 1. Setup
import tensorflow as tf
from tensorflow import keras
import numpy as np
import streamlit as st
import os, pickle, re

# 2. Functions to load the pickle objects and model
def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        pickle_object = pickle.load(f)
    return pickle_object

@st.cache_resource
def load_model(filepath):
    model_object = keras.models.load_model(filepath)
    return model_object

# 3. Define the filepath
label_encoder_filepath = 'label_encoder.pkl'
tokenizer_filepath = 'tokenizer.pkl'
model_filepath = 'model'

# 4. Load the label encoder, tokenizer and model
label_encoder = load_pickle(label_encoder_filepath)
tokenizer = load_pickle(tokenizer_filepath)
model = load_model(model_filepath)

# 5. Build the components of streamlit app
# a. A title
st.title("Ecommerce Product Classification")
# b. Text box for user input
with st.form('input_form'):
    text_area = st.text_area("Input your product description here")
    submitted = st.form_submit_button("Analyse the Text")

text_inputs = [text_area]

# c. Remove unwanted string from the text input
def remove_unwanted_string(text_inputs):
    for index, data in enumerate(text_inputs):
        text_inputs[index] = re.sub('<.*?>',' ', data)
        text_inputs[index] = re.sub("[^a-zA-Z]", " ", data).lower()
    return text_inputs

# Use the function to filter the unwanted string
text_filtered = remove_unwanted_string(text_inputs)
# Tokenize the string
text_token = tokenizer.texts_to_sequences(text_filtered)
# Padding and truncating
text_padded = keras.utils.pad_sequences(text_token, maxlen=200, padding='post', truncating='post')

# d. Use the model to make prediction
y_score = model.predict(text_padded)
y_pred = np.argmax(y_score, axis=1)

# e. Display the result
label_map = {i:classes for i, classes in enumerate(label_encoder.classes_)}
result = label_map[y_pred[0]]

# f. Write the prediction onto streamlit
st.header("Label list")
st.write(label_encoder.classes_)
st.header("Prediction score")
st.write(y_score)
st.header("Final prediction")
st.write(f"The type of news is: {result}")