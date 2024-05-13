import networkx as nx
import pandas as pd

import ast

movies = pd.read_csv("C:/Users/me/Downloads/WarlikeGreedyComputerscience 2/WarlikeGreedyComputerscience/tmdb_5000_movies.csv")
credits = pd.read_csv("C:/Users/me/Downloads/WarlikeGreedyComputerscience 2/WarlikeGreedyComputerscience/tmdb_5000_credits.csv")

movies.head()

movies = movies.merge(credits, on='title')

movies.head()

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.head()

movies.isnull().sum()

movies.dropna(inplace=True)

movies.duplicated().sum()


def convert(object):
    l = []
    for i in ast.literal_eval(object):
        l.append(i['name'])
    return l


movies['genres'] = movies['genres'].apply(convert)


def convert(object):
    l = []
    counter = 0
    for i in ast.literal_eval(object):
        if counter != 3:
            l.append(i['name'])
            counter += 1
        else:
            break
    return l


movies['cast'] = movies['cast'].apply(convert)


def fetch_director(object):
    l = []
    for i in ast.literal_eval(object):
        if i['job'] == 'Director':
            l.append(i['name'])
            break
    return l


movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies.head()
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['overview'] = movies['overview'].apply(lambda x: [i.replace(" ", "") for i in x])

movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew'] + movies['overview']

movies.head()

movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

movies.head()

import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


movies['tags'] = movies['tags'].apply(stem)

movies.head()

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(movies['tags']).toarray()

feature_names = cv.get_feature_names_out()  # Use this method for older versions of scikit-learn

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x: x[1])[1:6]


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        print(movies.iloc[i[0]].title)


recommend('Batman Begins')

import networkx as nx

# Create empty graph
G = nx.Graph()

# Add movies as nodes
movies = ["The Shawshank Redemption", "The Godfather", "The Dark Knight", "The Godfather: Part II", "12 Angry Men"]
G.add_nodes_from(movies)

# Add edges between similar movies
G.add_edge("The Shawshank Redemption", "The Godfather")
G.add_edge("The Shawshank Redemption", "The Dark Knight")
G.add_edge("The Shawshank Redemption", "The Godfather: Part II")
G.add_edge("The Godfather", "The Godfather: Part II")
G.add_edge("The Godfather", "12 Angry Men")
G.add_edge("The Dark Knight", "12 Angry Men")

# Recommend movies similar to a given movie
movie = "The Godfather"
recommendations = list(G.neighbors(movie))
print("Recommended movies for {}: {}".format(movie, recommendations))
