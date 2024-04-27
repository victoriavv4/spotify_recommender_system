
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import os


NUMERIC_FEATURES=['year', 'pop_bin',
       'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
       'duration_ms', 'time_signature']

class DataTransformer:
    def __init__(self):
        #self.year_encoder = OrdinalEncoder()
        self.scaler = MinMaxScaler()
        self.numeric_features = NUMERIC_FEATURES
        self.bin_edges = np.linspace(0, 100, 20)

    def ohe_feature(self, df, col, prefix):
        x = pd.get_dummies(df[col], prefix=prefix)
        return x

    def bin_popularity(self, df):
        df['pop_bin'] = np.searchsorted(self.bin_edges, df['popularity'])
        df.drop('popularity', axis=1, inplace=True)
        return df


    def fit_transform(self, df):

        df = self.bin_popularity(df)
        df_genre = self.ohe_feature(df, 'genre', 'genre')

        # convert to integer
        df_genre = df_genre.astype(int)

        df_numeric = pd.DataFrame(self.scaler.fit_transform(df[self.numeric_features].values),
                                  columns=self.numeric_features)

        #df_year = pd.DataFrame(self.year_encoder.fit_transform(df['year']))
        df_id = df[['track_id']]

        final_df = pd.concat([df_id, df_genre, df_numeric], axis=1)

        return final_df

    def transform(self, new_df):
        # Apply same transformation steps on new dataframe

        new_df = self.bin_popularity(new_df)
        # Scale numeric features using fitted Scaler
        #new_df[self.numeric_features] = new_df[numeric_features].apply(self.standard_scaler.transform)
        new_df[self.numeric_features] = self.scaler.transform(new_df[self.numeric_features].values)

        # remove the useless columns
        new_df = new_df.drop('Unnamed: 0', axis=1, errors='ignore')

        return new_df


class Clusterer:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        # self.kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        self.kmeans = KMeans(n_clusters=num_clusters, n_init=10)
        self.cluster_labels = None

    def fit(self, df):
        df_cluster = df.copy()
        df_cluster = df_cluster.drop("track_id", axis=1)
        self.kmeans.fit(df_cluster)
        self.cluster_labels = self.kmeans.labels_
        df['clusters'] = self.cluster_labels
        return df

    def assign_cluster(self, new_vector):
        cluster = self.kmeans.predict([new_vector])[0]
        return cluster

class MusicRecommender:
    def __init__(self, transformer, clusterer, clustered_main_df):
        self.transformer = transformer
        self.clusterer = clusterer
        self.clustered_main_df = clustered_main_df

    def keep_similarity_features(self, df):
        df= df.drop(['track_id','clusters'], axis=1)
        return df.values

    def cluster_finder(self, df, new_song_vector):
        return self.clusterer.assign_cluster(new_song_vector)

    def keep_related_cluster(self, df, cluster):
        return df[df['clusters']==cluster]

    def cosign_calculator(self, main_df_cluster_vectors, new_song_vector):
        return cosine_similarity(new_song_vector.reshape(1,-1), main_df_cluster_vectors)

    def recommend_similar_songs(self, new_df):

        # Transform the new song dataframe using the DataTransformer
        new_song_vector = self.transformer.transform(new_df).values

        # assign the new song to one cluster of the main dataframe
        related_cluster = self.cluster_finder(self.clustered_main_df, new_song_vector[0])

        # extract the assigned to the new song cluster from the main dataframe
        df_cosign = self.keep_related_cluster(self.clustered_main_df, related_cluster)

        # Get the vectors of all songs in the related cluster of the main dataframe
        df_cosign_vectors = self.keep_similarity_features(df_cosign)
        # main_df_vectors = self.transformed_main_df.values

        #check if the new song vector is in the main dataframe
        df_cosign_vectors = [v for v in df_cosign_vectors if not np.array_equal(v, new_song_vector)]

        # Calculate cosine similarity between the new song and all songs in the main dataframe
        similarities = self.cosign_calculator(df_cosign_vectors, new_song_vector)

        # Get the indices of the 10 most similar songs
        similar_song_indices = np.argsort(similarities.ravel())[::-1][:10]

        # Get the details of the 10 most similar songs
        similar_songs = pd.DataFrame(self.clustered_main_df.iloc[similar_song_indices]['track_id'])

        return similar_songs


def run_recommendor(new_df):
    with open('transformer_v2.pkl', "rb") as file:
        transformer = pickle.load(file)
        print("transformer loaded successfully...")

    with open('clusterer.pkl', "rb") as file:
        clusterer = pickle.load(file)
        print("clusterer loaded successfully...")

    with open('clustered_main_df.pkl', "rb") as file:
        clustered_main_df = pickle.load(file)
        print("the clustered main dataframe loaded successfully...")

    recommender = MusicRecommender(transformer, clusterer, clustered_main_df)
    similar_songs = recommender.recommend_similar_songs(new_df)
    return similar_songs

# new_df=pd.read_csv(r"track_vec 3.csv")

# if __name__=="__main__":

    # df_spotify = pd.read_csv('spotify_data.csv')

    # transformer = DataTransformer()
    # clusterer = Clusterer(N_CLUSTERS)
    # transformed_main_df = transformer.fit_transform(df_spotify)
    # clustered_main_df = clusterer.fit(transformed_main_df)

    # print("clustered_main_df.head()")
    # print(clustered_main_df.head())
    # print("Cluster Counts")
    # print(clustered_main_df['clusters'].value_counts())

    # with open(r'clustered_main_df.pkl', 'wb') as f1:
    #     pickle.dump(clustered_main_df, f1)
    #     f1.close()
    #     print("transformed main df pickled...")
    # with open(r'transformer_v2.pkl', 'wb') as f2:
    #     pickle.dump(transformer, f2)
    #     f2.close()
    #     print("transformer pickled...")
    # with open(r'clusterer.pkl', 'wb') as f3:
    #     pickle.dump(clusterer, f3)
    #     f3.close()
    #     print("clusterer pickled...")

    # recommended_songs=run_recommendor(new_df)
    # print("recommeded songs:")
    # print()
    # print(recommended_songs)
    # recommended_songs.to_csv(r'clustering_recommended-songs_5.csv')



