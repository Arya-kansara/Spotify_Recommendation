import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import os


# Get the directory where this script (spotify_recommendation.py) is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create the full path to the CSV file dynamically
csv_path = os.path.join(current_dir, "spotify_dataset.csv")

# Load the dataset using the dynamic path
df = pd.read_csv(csv_path)df.fillna("Unknown", inplace=True)

# Feature Engineering
df['release_year'] = pd.to_datetime(df['Release Date'], errors='coerce').dt.year
df['release_year'].fillna(df['release_year'].median(), inplace=True)

le_genre = LabelEncoder()
le_artist = LabelEncoder()
le_album = LabelEncoder()

df['genre_enc'] = le_genre.fit_transform(df['genre'])
df['artist_enc'] = le_artist.fit_transform(df['artist'])
df['album_enc'] = le_album.fit_transform(df['album'])

df['artist_song_count'] = df.groupby('artist')['artist'].transform('count')
df['album_song_count'] = df.groupby('album')['album'].transform('count')

features = [
    'genre_enc',
    'artist_enc',
    'album_enc',
    'artist_song_count',
    'album_song_count',
    'release_year'
]

X = df[features]
y = df['popularity']

# Models
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X, y)

def ensemble_predict(X):
    return (rf.predict(X) + xgb.predict(X)) / 2

# 🎵 NEXT SONG PREDICTION FUNCTION
def next_song_prediction(track_name):
    track_name = track_name.lower()

    current_song = df[df['track_name'].str.lower() == track_name]

    if current_song.empty:
        return None

    current_song = current_song.iloc[0]

    # Step 1: Same artist & album
    candidates = df[
        (df['artist'] == current_song['artist']) &
        (df['album'] == current_song['album']) &
        (df['track_name'] != current_song['track_name'])
    ]

    # Step 2: Same artist
    if candidates.empty:
        candidates = df[
            (df['artist'] == current_song['artist']) &
            (df['track_name'] != current_song['track_name'])
        ]

    # Step 3: Same genre
    if candidates.empty:
        candidates = df[
            (df['genre'] == current_song['genre']) &
            (df['track_name'] != current_song['track_name'])
        ]

    if candidates.empty:
        return None

    candidates = candidates.copy()
    candidates['score'] = ensemble_predict(candidates[features])

    next_song = candidates.sort_values(
        by=['score', 'popularity'], ascending=False
    ).iloc[0]

    return {
        'track_name': next_song['track_name'],
        'artist': next_song['artist'],
        'album': next_song['album'],
        'genre': next_song['genre']
    }

