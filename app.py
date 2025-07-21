import streamlit as st
import pickle 
import pandas as pd
import numpy as np

st.title('Movie Recommendation System')

# Load pickled files
dataset = pickle.load(open('movies.pkl','rb'))
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(dataset['tags']).toarray()
similarity = cosine_similarity(vector)
# Get movie titles
movies_list = dataset['title_x'].values
option = st.selectbox('Select a movie:', movies_list)

# Recommend function
def recommend(movie_name):
    index = dataset[dataset['title_x'] == movie_name].index[0]
    movie_list = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = []
    for i in movie_list:
        recommended_movies.append(dataset.iloc[i[0]].title_x)
    return recommended_movies

# Button
if st.button("Recommend"):
    recommendations = recommend(option)
    for i in recommendations:
        st.write(i)
else:
    st.write("Select a movie and press recommend.")
    
