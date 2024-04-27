
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


NUMERIC_FEATURES=['popularity', 'year',
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

        # Scale numeric features using fitted Scaler
        #new_df[self.numeric_features] = new_df[numeric_features].apply(self.standard_scaler.transform)
        new_df[self.numeric_features] = self.scaler.transform(new_df[self.numeric_features].values)

        # remove the useless columns
        # new_df = new_df.drop('Unnamed: 0', axis=1)
        new_df = new_df.drop('Unnamed: 0', axis=1, errors='ignore')

        return new_df


class MusicRecommender:
    def __init__(self, transformer, transformed_main_df):
        self.transformer = transformer
        self.transformed_main_df = transformed_main_df


    def keep_similarity_features(self, df):
        df= df.drop('track_id', axis=1)
        return df.values

    def cosign_calculator(self, main_df_vectors, new_song_vector):
        return cosine_similarity(new_song_vector.reshape(1,-1), main_df_vectors)

    def recommend_similar_songs(self, new_df):
        # Transform the new dataframe using the DataTransformer
        new_song_vector = self.transformer.transform(new_df).values

        # Get the vectors of the new song
        # new_song_vector = self.keep_similarity_features(transformed_new_df)

        # Get the vectors of all songs in the main dataframe
        main_df_vectors = self.keep_similarity_features(self.transformed_main_df)
        # main_df_vectors = self.transformed_main_df.values

        #check if the new song vector is in the main dataframe
        main_df_vectors = [v for v in main_df_vectors if not np.array_equal(v, new_song_vector)]

        # Calculate cosine similarity between the new song and all songs in the main dataframe
        similarities = self.cosign_calculator(main_df_vectors, new_song_vector)


        # Get the indices of the 10 most similar songs
        similar_song_indices = np.argsort(similarities.ravel())[::-1][:10]


        # Get the details of the 10 most similar songs
        similar_songs = pd.DataFrame(self.transformed_main_df.iloc[similar_song_indices]['track_id'])
        return similar_songs


def run_recommendor(new_df):
    with open(r'../../Desktop/OneDrive_1_2024-03-29/transformer.pkl', "rb") as file:
        transformer=pickle.load(file)
        print("transformer loaded successfully...")

    with open(r'transformed_main_df.pkl', "rb") as file:
        transformed_main_df=pickle.load(file)
        print("the transformed main dataframe loaded successfully...")

    recommender = MusicRecommender(transformer, transformed_main_df)
    similar_songs = recommender.recommend_similar_songs(new_df)

    return similar_songs

# new_df = pd.read_csv("track_vec 3.csv")

# if __name__=="__main__":

#     df_spotify = pd.read_csv('spotify_data.csv')
#     transformer = DataTransformer()
#     transformed_main_df = transformer.fit_transform(df_spotify)
#     with open(r'transformed_main_df.pkl', 'wb') as f1:
#         pickle.dump(transformed_main_df, f1)
#         f1.close()
#         print("transformed main df pickled...")
#     with open(r'transformer.pkl', 'wb') as f2:
#         pickle.dump(transformer, f2)
#         f2.close()
#         print("transformer pickled...")

#     recommended_songs = run_recommendor(new_df)
#     print("recommended songs:")
#     print(recommended_songs)



