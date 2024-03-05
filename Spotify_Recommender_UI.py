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

import Spotify_Recommender_Functions

# get spotify dataframe
spotify_df = pd.read_csv('spotify_data.csv')

# Connect to Spotify

# Victoria's Credentials
# CLIENT_ID = ''
# CLIENT_SECRET = ''
# REDIRECT_URI = ''



# Initialize the Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Sidebar
st.sidebar.header('Username Playlists')
st.sidebar.subheader('Username')
username = st.sidebar.text_input("Enter your Spotify username:")
button_clicked = st.sidebar.button("Get Playlists")

# Sidebar Header
st.sidebar.subheader('Playlists')

# Header
st.header('Spotify User Recommendation System')

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
                    st.write(f"Artist: {artist}")
                    st.write(f"Album: {album}")
                    st.write(f"Release Year: {release_year}")
                    st.image(image_url, caption='Album Cover', width=200)

                    # Get track features as a JSON file
                    if track_id is not None:
                        track_features_df = Spotify_Recommender_Functions.track_vector(track_id, sp, spotify_df)
                        # st.dataframe(track_features_df)

                        # Get recommendations
                        st.write("")
                        st.subheader("Recommended Songs for:")
                        st.subheader(f"{name}")

                        token = Spotify_Recommender_Functions.get_token(CLIENT_ID, CLIENT_SECRET)
                        similar_songs_json = Spotify_Recommender_Functions.get_recommendations(track_id, token, 10)
                        df_recommendations = pd.DataFrame(similar_songs_json['tracks'])[['name']]

                        recommended_tracks = []
                        for i in range(len(df_recommendations)):
                            recommended_tracks.append(df_recommendations.iloc[i, 0])

                        st.write("")
                        selected_recommendation = st.selectbox("Select a recommendation:", recommended_tracks)

                        if selected_recommendation is not None:
                            sp_recommendation = sp.search(q=selected_recommendation, type='track', limit=1)
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
                                st.write(f"Artist: {artist}")
                                st.write(f"Album: {album}")
                                st.write(f"Release Year: {release_year}")
                                st.image(image_url, caption='Album Cover', width=500)




    except spotipy.SpotifyException as e:
        st.error(f"Error: {e}")
