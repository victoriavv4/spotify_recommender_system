import pandas as pd
import seaborn as sns
import re
import sys
import numpy as np
import spotipy
import spotipy.util as util
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
from skimage import io
from sklearn.metrics.pairwise import cosine_similarity

import Spotify_Recommender_Functions_Beta
import Spotify_Recommender_Cosign_Beta
import Spotify_Recommender_KMeans_Cosign_Beta
import os

# Select the number of columns
col1, col2, col3 = st.columns(3)

# Set the LOKY_MAX_CPU_COUNT environment variable
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

# get spotify dataframe
df_spotify = pd.read_csv('spotify_data.csv')

# Only Run this code only once
# Pre-process the Spotify Track List
# Pickle the pre-processed dataframe

# Cosign Only Files
# transformer = Spotify_Recommender_Cosign_Beta.DataTransformer()
# transformed_main_df = transformer.fit_transform(df_spotify)
# with open(r'transformed_main_df.pkl', 'wb') as f1:
#      pickle.dump(transformed_main_df, f1)
#      f1.close()
#      print("transformed main df pickled...")
# with open(r'transformer.pkl', 'wb') as f2:
#      pickle.dump(transformer, f2)
#      f2.close()
#      print("transformer pickled...")
#
# # K-Means-Cosign Only Files
# transformer = Spotify_Recommender_KMeans_Cosign_Beta.DataTransformer()
# transformed_main_df = transformer.fit_transform(df_spotify)
# N_CLUSTERS=5
# clusterer = Spotify_Recommender_KMeans_Cosign_Beta.Clusterer(N_CLUSTERS)
# clustered_main_df = clusterer.fit(transformed_main_df)
# with open('clustered_main_df.pkl', 'wb') as f3:
#      pickle.dump(clustered_main_df, f3)
#      f3.close()
#      print("clustered main df pickled...")
# with open('transformer_v2.pkl', 'wb') as f4:
#      pickle.dump(transformer, f4)
#      f4.close()
#      print("transformer_v2 pickled...")
# with open('clusterer.pkl', 'wb') as f5:
#      pickle.dump(clusterer, f5)
#      f5.close()
#      print("clusterer pickled...")

# Connect to Spotify

# Victoria's Credentials
CLIENT_ID = '#'
CLIENT_SECRET = '#'
REDIRECT_URI = '#'

# Patrick's Credentials
#CLIENT_ID = '#'
#CLIENT_SECRET = '#'
#REDIRECT_URI = '#'

# Initialize the Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Sidebar
st.sidebar.header('Username Playlists')
st.sidebar.subheader('Username')
username = st.sidebar.text_input("Enter your Spotify username:")
button_clicked = st.sidebar.button("Get Playlists")

# For Testing
# username = '#'
# username = '#'

# Sidebar Header
st.sidebar.subheader('Playlists')

# Header
st.header('Spotify Track Recommender System')


if username:
    try:
        # Retrieve user playlists
        playlists = sp.user_playlists(username)

        # Display user playlists & Get Selected Playlist
        selected_playlist_name = st.sidebar.selectbox("Select a playlist:",
                                                      [playlist['name'] for playlist in playlists['items']])
        selected_playlist = next(
            (playlist for playlist in playlists['items'] if playlist['name'] == selected_playlist_name), None)

        # Select Song
        if selected_playlist:
            # Display track names
            tracks = {}
            playlist_tracks = sp.playlist_items(selected_playlist['id'])['items']
            for idx, item in enumerate(playlist_tracks):
                tracks[item['track']['name']] = item['track']['id']

            selected_track = st.selectbox("Select a track:", list(tracks.keys()))

            if selected_track:

                # Perform song search
                track_json = sp.track(tracks[selected_track])

                if track_json:
                    # track_info = sp_track['tracks']['items'][0]
                    track_id = tracks[selected_track]
                    name = track_json['name']
                    artist = track_json['artists'][0]['name']
                    album = track_json['album']['name']
                    release_year = track_json['album']['release_date'][:4] if 'release_date' in track_json[
                        'album'] else 'N/A'
                    image_url = track_json['album']['images'][0]['url'] if track_json['album']['images'] else 'N/A'

                    # Display track information in the frontend
                    st.write(f"Release Year: {release_year}")
                    st.image(image_url, caption='Album Cover', width=200)
                    st.write(f"Artist: {artist}")
                    st.write(f"Album: {album}")
                    st.write("")

                    # Get track features as a JSON file
                    if track_id is not None:

                        # Create Track Vector
                        track_features_df = Spotify_Recommender_Functions_Beta.track_vector(track_id, sp, df_spotify)

                        # Cosign Similar Songs
                        # --------------------

                        # Get Cosign Recommended Songs
                        cosign_recommended_songs = Spotify_Recommender_Cosign_Beta.run_recommendor(track_features_df)

                        # Create list of Cosign Recommended Songs
                        cosign_recommended_songs_list = []
                        for id in cosign_recommended_songs['track_id']:
                            cosign_recommended_songs_list.append(df_spotify[df_spotify['track_id'] == id]['track_name'].iloc[0])

                        # Get Clustered - Cosign Recommended Songs
                        clustered_cosign_recommended_songs = Spotify_Recommender_KMeans_Cosign_Beta.run_recommendor(track_features_df)

                        # Create list of Clustered Cosign Recommended Songs
                        clustered_cosign_recommended_songs_list = []
                        for id in clustered_cosign_recommended_songs['track_id']:
                            clustered_cosign_recommended_songs_list.append(df_spotify[df_spotify['track_id'] == id]['track_name'].iloc[0])

                        # Get recommendations
                        # col1, col2, col3 = st.columns(3, gap = "medium")
                        col1, col2 = st.columns(2, gap="medium")
                        st.write("")

                        with col1:
                            st.subheader("Cosine Similarity Only")
                            # st.subheader(f"{name}")

                            st.write("")
                            selected_cosign_recommendation = st.selectbox("Select a recommended track:", cosign_recommended_songs_list)

                            if selected_cosign_recommendation is not None:
                                sp_recommendation = sp.search(q=selected_cosign_recommendation, type='track', limit=1)
                                if sp_recommendation['tracks']['items']:
                                    track_info = sp_recommendation['tracks']['items'][0]
                                    track_id = track_info['id']
                                    name = track_info['name']
                                    artist = track_info['artists'][0]['name']
                                    album = track_info['album']['name']
                                    release_year = track_info['album']['release_date'][:4] if 'release_date' in track_info[
                                        'album'] else 'N/A'
                                    image_url = track_info['album']['images'][0]['url'] if track_info['album'][
                                        'images'] else 'N/A'

                                    # Display track information in the frontend
                                    st.write("")
                                    st.write(f"Release Year: {release_year}")
                                    st.image(image_url, caption='Album Cover', width=200)
                                    st.write(f"Artist: {artist}")
                                    st.write(f"Album: {album}")

                        with col2:
                            st.subheader("Hybrid: KMeans - Cosine")
                            # st.subheader(f"{name}")

                            st.write("")
                            selected_clustered_cosign_recommendation = st.selectbox("Select a recommended track:", clustered_cosign_recommended_songs_list)

                            if selected_clustered_cosign_recommendation is not None:
                                sp_recommendation = sp.search(q=selected_clustered_cosign_recommendation, type='track', limit=1)
                                if sp_recommendation['tracks']['items']:
                                    track_info = sp_recommendation['tracks']['items'][0]
                                    track_id = track_info['id']
                                    name = track_info['name']
                                    artist = track_info['artists'][0]['name']
                                    album = track_info['album']['name']
                                    release_year = track_info['album']['release_date'][:4] if 'release_date' in track_info[
                                        'album'] else 'N/A'
                                    image_url = track_info['album']['images'][0]['url'] if track_info['album'][
                                        'images'] else 'N/A'

                                    # Display track information in the frontend
                                    st.write("")
                                    st.write(f"Release Year: {release_year}")
                                    st.image(image_url, caption='Album Cover', width=200)
                                    st.write(f"Artist: {artist}")
                                    st.write(f"Album: {album}")


                        # Spotify Recommended Songs
                        # -------------------------

                        # token = Spotify_Recommender_Functions_Beta.get_token(CLIENT_ID, CLIENT_SECRET)
                        # similar_songs_json = Spotify_Recommender_Functions_Beta.get_recommendations(track_id, token, 10)
                        # df_recommendations = pd.DataFrame(similar_songs_json['tracks'])[['name']]

                        # recommended_tracks = []
                        # for i in range(len(df_recommendations)):
                        #     recommended_tracks.append(df_recommendations.iloc[i, 0])

                        # with col3:
                        #     st.subheader("Spotify")
                        #     # st.subheader(f"{name}")
                        #     st.write("")
                        #     selected_recommendation = st.selectbox("Select a recommended track:", recommended_tracks)

                        #     if selected_recommendation is not None:
                        #         sp_recommendation = sp.search(q=selected_recommendation, type='track', limit=1)
                        #         if sp_recommendation['tracks']['items']:
                        #             track_info = sp_recommendation['tracks']['items'][0]
                        #             track_id = track_info['id']
                        #             name = track_info['name']
                        #             artist = track_info['artists'][0]['name']
                        #             album = track_info['album']['name']
                        #             release_year = track_info['album']['release_date'][:4] if 'release_date' in track_info[
                        #                 'album'] else 'N/A'
                        #             image_url = track_info['album']['images'][0]['url'] if track_info['album'][
                        #                 'images'] else 'N/A'

                                    # Display track information in the frontend
                        #             st.write("")
                        #             st.write(f"Release Year: {release_year}")
                        #             st.image(image_url, caption='Album Cover', width=200)
                        #             st.write(f"Artist: {artist}")
                        #             st.write(f"Album: {album}")



    except spotipy.SpotifyException as e:
        st.error(f"Error: {e}")
