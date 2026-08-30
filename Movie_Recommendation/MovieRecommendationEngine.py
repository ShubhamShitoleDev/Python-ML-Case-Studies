#################################################################################
#
#  Project Name  : Movie Recommendation Engine
#  Description   : Loads the MovieLens dataset and builds three
#                   recommendation strategies:
#                     1. Content-Based Filtering — recommends movies with
#                        similar genres using cosine similarity.
#                     2. Clustering-Based Filtering — groups movies into
#                        genre clusters using K-Means, then recommends
#                        movies from the same cluster.
#                     3. Collaborative Filtering — recommends movies based
#                        on what similar users (by rating pattern) enjoyed.
#  Date          : 23-Aug-2026
#  Author        : Shubham Namdev Shitole
#
#################################################################################

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

Border = "-"*50


#################################################################################
#
# Function Name : load_data
# Input :         movies_path, ratings_path
# Description :   Loads the MovieLens movies and ratings CSV files and
#                 prints an EDA summary of each
# Return Value :  movies DataFrame, ratings DataFrame
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def load_data(movies_path, ratings_path):
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    print("Movies Shape : ", movies.shape)
    print("Movies Sample : ")
    print(movies.head())

    print(Border)
    print("Ratings Shape : ", ratings.shape)
    print("Ratings Sample : ")
    print(ratings.head())

    print(Border)
    print("Missing Values (movies) : ")
    print(movies.isnull().sum())
    print("Missing Values (ratings) : ")
    print(ratings.isnull().sum())

    return movies, ratings


#################################################################################
#
# Function Name : build_genre_vectors
# Input :         movies
# Description :   Converts each movie's pipe-separated genre string (e.g.
#                 "Action|Adventure|Sci-Fi") into a binary genre vector,
#                 one column per unique genre, using CountVectorizer. This
#                 numeric representation is what both the similarity and
#                 clustering approaches are built on.
# Return Value :  genre_vectors (numpy array, one row per movie)
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def build_genre_vectors(movies):
    # CountVectorizer splits on whitespace by default, so genres are
    # rejoined with spaces instead of "|" before vectorizing.
    genre_text = movies["genres"].str.replace("|", " ", regex=False)

    vectorizer = CountVectorizer(token_pattern=r"[^\s]+")
    genre_vectors = vectorizer.fit_transform(genre_text).toarray()

    print("Unique genres found : ", list(vectorizer.get_feature_names_out()))
    print("Genre vector shape : ", genre_vectors.shape)

    return genre_vectors


#################################################################################
#
# Function Name : content_based_recommend
# Input :         movie_title, movies, genre_vectors, top_n
# Description :   Finds the given movie's genre vector, computes cosine
#                 similarity against every other movie's genre vector, and
#                 returns the most similar movies by genre overlap
# Return Value :  DataFrame of the top_n most similar movie titles
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def content_based_recommend(movie_title, movies, genre_vectors, top_n=5):
    matches = movies[movies["title"].str.contains(movie_title, case=False, na=False)]

    if matches.empty:
        print(f"No movie found matching '{movie_title}'")
        return None

    movie_index = matches.index[0]
    matched_title = movies.loc[movie_index, "title"]
    print(f"Finding movies similar to : {matched_title}")

    similarity_scores = cosine_similarity(
        genre_vectors[movie_index].reshape(1, -1), genre_vectors
    )[0]

    similar_indices = similarity_scores.argsort()[::-1]
    # Exclude the movie itself (always the highest similarity, score = 1.0)
    similar_indices = [i for i in similar_indices if i != movie_index][:top_n]

    recommendations = movies.iloc[similar_indices][["title", "genres"]].copy()
    recommendations["similarity"] = similarity_scores[similar_indices]

    return recommendations


#################################################################################
#
# Function Name : cluster_movies_by_genre
# Input :         movies, genre_vectors, n_clusters
# Description :   Groups movies into n_clusters using K-Means on their
#                 genre vectors, so movies with overlapping genre profiles
#                 end up in the same cluster
# Return Value :  movies DataFrame with a new "GenreCluster" column
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def cluster_movies_by_genre(movies, genre_vectors, n_clusters=10):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = model.fit_predict(genre_vectors)

    clustered_movies = movies.copy()
    clustered_movies["GenreCluster"] = cluster_labels

    print(f"Formed {n_clusters} genre clusters")
    print("Movies per cluster : ")
    print(clustered_movies["GenreCluster"].value_counts().sort_index())

    return clustered_movies


#################################################################################
#
# Function Name : cluster_based_recommend
# Input :         movie_title, clustered_movies, top_n
# Description :   Recommends movies that fall in the same K-Means genre
#                 cluster as the given movie
# Return Value :  DataFrame of top_n movie titles from the same cluster
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def cluster_based_recommend(movie_title, clustered_movies, top_n=5):
    matches = clustered_movies[clustered_movies["title"].str.contains(movie_title, case=False, na=False)]

    if matches.empty:
        print(f"No movie found matching '{movie_title}'")
        return None

    matched_row = matches.iloc[0]
    cluster_id = matched_row["GenreCluster"]
    print(f"'{matched_row['title']}' belongs to Genre Cluster {cluster_id}")

    same_cluster = clustered_movies[
        (clustered_movies["GenreCluster"] == cluster_id) &
        (clustered_movies["title"] != matched_row["title"])
    ]

    return same_cluster[["title", "genres"]].head(top_n)


#################################################################################
#
# Function Name : build_user_item_matrix
# Input :         ratings
# Description :   Pivots the ratings table into a User x Movie matrix, where
#                 each cell is the rating that user gave that movie (0 for
#                 movies they haven't rated)
# Return Value :  user_item_matrix (DataFrame, rows = userId, columns = movieId)
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def build_user_item_matrix(ratings):
    user_item_matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)

    print("User-Item Matrix shape : ", user_item_matrix.shape)
    print("(rows = users, columns = movies, 0 = not rated)")

    return user_item_matrix


#################################################################################
#
# Function Name : collaborative_recommend
# Input :         user_id, user_item_matrix, movies, top_n
# Description :   Finds users with similar rating patterns to the given
#                 user (cosine similarity across their rating vectors),
#                 then recommends movies those similar users rated highly
#                 that the target user hasn't seen yet
# Return Value :  DataFrame of top_n recommended movie titles
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def collaborative_recommend(user_id, user_item_matrix, movies, top_n=5):
    if user_id not in user_item_matrix.index:
        print(f"No user found with id {user_id}")
        return None

    similarity_scores = cosine_similarity(user_item_matrix)
    user_position = user_item_matrix.index.get_loc(user_id)

    similar_user_scores = similarity_scores[user_position]
    similar_user_positions = similar_user_scores.argsort()[::-1]
    # Exclude the user themself (always highest similarity, score = 1.0)
    similar_user_positions = [i for i in similar_user_positions if i != user_position][:10]

    similar_users = user_item_matrix.iloc[similar_user_positions]

    # Average rating given by similar users, for movies the target user
    # has NOT already rated
    target_user_ratings = user_item_matrix.loc[user_id]
    unseen_movies = target_user_ratings[target_user_ratings == 0].index

    predicted_scores = similar_users[unseen_movies].mean(axis=0).sort_values(ascending=False)
    top_movie_ids = predicted_scores.head(top_n).index

    recommendations = movies[movies["movieId"].isin(top_movie_ids)][["movieId", "title", "genres"]]

    return recommendations


#################################################################################
#
# Function Name : main
# Input :         None
# Description :   Runs the full Movie Recommendation Engine pipeline:
#                 load data, build genre vectors, demonstrate content-based
#                 recommendations, cluster movies by genre and demonstrate
#                 cluster-based recommendations, then build the user-item
#                 matrix and demonstrate collaborative filtering
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def main():
    print(Border)
    print("Step 1 : Load the Data Set")
    print(Border)

    movies, ratings = load_data("movies.csv", "ratings.csv")

    print(Border)
    print("Step 2 : Build Genre Vectors")
    print(Border)

    genre_vectors = build_genre_vectors(movies)

    print(Border)
    print("Step 3 : Content-Based Recommendation (Similarity)")
    print(Border)

    sample_movie = movies["title"].iloc[0]
    print(f"\nExample recommendation for : {sample_movie}")
    print(content_based_recommend(sample_movie, movies, genre_vectors, top_n=5))

    print(Border)
    print("Step 4 : Clustering-Based Recommendation (K-Means on Genres)")
    print(Border)

    clustered_movies = cluster_movies_by_genre(movies, genre_vectors, n_clusters=10)

    print(f"\nExample cluster-based recommendation for : {sample_movie}")
    print(cluster_based_recommend(sample_movie, clustered_movies, top_n=5))

    print(Border)
    print("Step 5 : Build User-Item Matrix")
    print(Border)

    user_item_matrix = build_user_item_matrix(ratings)

    print(Border)
    print("Step 6 : Collaborative Filtering Recommendation")
    print(Border)

    sample_user_id = user_item_matrix.index[0]
    print(f"\nExample recommendation for User {sample_user_id} (based on similar users) : ")
    print(collaborative_recommend(sample_user_id, user_item_matrix, movies, top_n=5))


if __name__ == "__main__":
    main()
