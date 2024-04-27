import spotipy
import pandas as pd
import requests

# Function to create a dataframe from a song containing the same features as the spotify list
def track_vector(track_id, sp, spotify_df):

    # extracting audio_features of the track
    metadata_list = []
    audio_ft = sp.audio_features(track_id)

    #remove irrelevant audio features from the dictionary
    rem_keys = ['type', 'id', 'uri', 'track_href', 'analysis_url']
    clean_audio_ft = [{key: value for key, value in d.items() if key not in rem_keys} for d in audio_ft]

    metadata_list.append(clean_audio_ft)

    flat_meta_data = [item for sublist in metadata_list for item in sublist]
    user_playlist_df = pd.DataFrame(flat_meta_data)

    # create year & popularity column
    # user_playlist_df['artist_name'] = sp.track(track_id)['album']['artists'][0]['name']
    user_playlist_df['popularity'] = sp.track(track_id)['popularity']
    user_playlist_df['year'] = sp.track(track_id)['album']['release_date'][:4]

    # get artists genres
    track_data = sp.track(track_id)
    artist_ids = track_data['album']['artists'][0]['id']
    artist_genres = sp.artist(artist_ids)['genres']
    genres_str =  ', '.join(artist_genres) if artist_genres else ''

    # create matching genre columns from tf-idf
    genres_list = [genre.replace('indie-pop', 'indie') for genre in spotify_df['genre'].unique()]
    for g in genres_list:
        user_playlist_df['genre_' + g] = int(any(word.lower()
        in genres_str.replace(',', '').lower().split() for word in g.split()))

    # slicing df
    genre_cols = [c for c in user_playlist_df.columns if c.startswith('genre_')]
    genre_df = user_playlist_df.loc[:, genre_cols]

    other_cols = ['popularity', 'year',
       'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
       'duration_ms', 'time_signature']

    other_df = user_playlist_df.loc[:, other_cols]
    user_playlist_df = pd.concat([genre_df, other_df], axis = 1)

    return user_playlist_df


# Temporary function to recommend similar tracks using Spotify Recommender
def get_token(CLIENT_ID, CLIENT_SECRET):
    # Get token
    auth_url = 'https://accounts.spotify.com/api/token'
    data = {
        'grant_type': 'client_credentials',
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
    }
    auth_response = requests.post(auth_url, data=data)
    token = auth_response.json().get('access_token')
    return token

def get_recommendations(track, token, k):
    limit = k
    recUrl = f"https://api.spotify.com/v1/recommendations?limit={limit}&seed_tracks={track}"
    headers = {
        "Authorization": "Bearer " + token
    }
    res = requests.get(url = recUrl, headers = headers)
    return res.json()






