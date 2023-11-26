import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.neighbors import KNeighborsClassifier

MODEL_URL = "https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/universal-sentence-encoder/versions/2"
sentence_encoder_layer = hub.KerasLayer(MODEL_URL, input_shape=[], dtype=tf.string, trainable=False, name="use")

@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

@st.cache_resource
def get_model():
    with open("model.pkl", 'rb') as file:
        model = pickle.load(file)
    return model

df = load_data()
model = get_model()
nn = model["model"]
st.title("Research Paper Recommender")

def calculate_user_embedding(title, abstract):
    user_abstract = sentence_encoder_layer([title, abstract])
    return user_abstract.numpy()

def find_similar_papers(user_embeddings, k=5):
    dist, indices = nn.kneighbors(X=user_embeddings, n_neighbors=k)
    return indices[0]

def main():
    with st.form("recommendation_form"):
        title = st.text_input('Please Enter the Paper Title')
        abstract = st.text_area('Please Enter the Abstract of the Paper')
        submit_button = st.form_submit_button("Recommend Papers!")
    if submit_button:
        user_embeddings = calculate_user_embedding(title, abstract)
        similar_paper_indices = find_similar_papers(user_embeddings)
        st.write("Recommendations are: ")
        for i, idx in enumerate(similar_paper_indices):
            recommended_paper = df['title'][idx]
            st.write(f"Recommendation {i + 1}:\n{recommended_paper}\n")
        
if __name__ == "__main__":
    main()
    

