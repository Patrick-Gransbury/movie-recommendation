from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import ast
import nltk
import re
import requests
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

nltk.download('stopwords')

TMDB_API_KEY = 'a1fe2f0ac92d2dd849674115d68777a5'

ps = PorterStemmer()
cv = CountVectorizer(max_features = 5000, stop_words = 'english')

credits_df = pd.read_csv('credits.csv')
movies_df = pd.read_csv('movies.csv')

movies_df = movies_df.merge(credits_df, on='title')
movies_df = movies_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'popularity']]
movies_df.dropna(inplace=True)

def convert_list(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def fetch_lead_actors(obj):
    L=[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
    return L

port_stem = PorterStemmer()

def stemming(content):
    # Only keeping alphabets in the content 
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    # Removing spaces and converting to lowercase
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    # Stemming the words that are not in stopwords
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    # Joining the list of words into a string
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

movies_df['overview'] = movies_df['overview'].apply(stemming)
movies_df['overview'] = movies_df['overview'].apply(lambda x:x.split())
movies_df['genres'] = movies_df['genres'].apply(convert_list)
movies_df['keywords'] = movies_df['keywords'].apply(convert_list)
movies_df['cast'] = movies_df['cast'].apply(fetch_lead_actors)
movies_df['crew'] = movies_df['crew'].apply(fetch_director)
movies_df['genres'] = movies_df['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies_df['cast'] = movies_df['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies_df['crew'] = movies_df['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['cast'] + movies_df['crew']

movies = movies_df[['movie_id', 'title', 'tags']]
movies['tags'] = movies['tags'].apply(lambda x:' '.join(x))
movies['tags'] = movies['tags'].apply(lambda x:x.lower())

vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x:x[1])[1:11]

def split_names(name):
    
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', name)

def fetch_poster_path(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US'
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f'https://image.tmdb.org/t/p/w500/{poster_path}'
    except requests.exceptions.RequestException as e:
        print(f"Error fetching poster for movie ID {movie_id}: {e}")
    return None

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:26]
    
    # Create a dictionary to store the combined scores (similarity + popularity_weight * popularity)
    combined_scores = {}

    for i in movies_list:
        index = i[0]
        
        # Define your weights
        popularity_weight = 0.002
        similarity_weight = 1-popularity_weight

        # Calculate the combined score
        combined_score = (similarity_weight * i[1]) + (popularity_weight * movies_df.iloc[index].popularity)

        combined_scores[index] = combined_score

    # Sort the dictionary by combined scores in descending order
    
    sorted_combined_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    # Get the top 10 movies based on the combined scores
    movies_final_list = sorted_combined_scores[:10]
    
    recommended_movies = []
    for i in movies_final_list:
        movie_data = {}
        index = i[0]
        movie_data['title'] = movies_df.iloc[index]['title']
        movie_data['overview'] = ' '.join(movies_df.iloc[index]['overview'])
        movie_data['cast'] = ', '.join(movies_df.iloc[index]['cast'])
        movie_data['crew'] = ', '.join(movies_df.iloc[index]['crew'])
        movie_data['poster_url'] = fetch_poster_path(movies_df.iloc[index]['movie_id'])
        recommended_movies.append(movie_data)
    
    return recommended_movies

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    try:
        data = request.get_json()
        movie = data.get('movie')
        if not movie:
            return jsonify({'error': 'Please provide a valid movie title.'}), 400
        
        recommendations = recommend(movie)
        if not recommendations:
            return jsonify({'error': 'Movie not found in the dataset.'}), 404
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)