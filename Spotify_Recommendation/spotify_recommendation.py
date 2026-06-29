import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import os

# --- 1. LOAD DATA SAFELY ---
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "spotify_dataset.csv")

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print("❌ Error: csv file not found. Make sure spotify_dataset.csv is in the same folder.")
    df = pd.DataFrame()

# --- 2. TRAIN LIGHTER MODELS (CRITICAL FIX) ---
if not df.empty:
    df.fillna("Unknown", inplace=True)
    
    # Simple feature engineering
    df['release_year'] = pd.to_datetime(df['Release Date'], errors='coerce').dt.year
    df['release_year'] = df['release_year'].fillna(df['release_year'].median())

    le_genre = LabelEncoder()
    le_artist = LabelEncoder()
    le_album = LabelEncoder()

    df['genre_enc'] = le_genre.fit_transform(df['genre'])
    df['artist_enc'] = le_artist.fit_transform(df['artist'])
    df['album_enc'] = le_album.fit_transform(df['album'])

    df['artist_song_count'] = df.groupby('artist')['artist'].transform('count')
    df['album_song_count'] = df.groupby('album')['album'].transform('count')

    features = ['genre_enc', 'artist_enc', 'album_enc', 'artist_song_count', 'album_song_count', 'release_year']
    X = df[features]
    y = df['popularity']

    # ⚠️ OPTIMIZATION: Reduced n_estimators from 100 to 15 to prevent crashing
    rf = RandomForestRegressor(n_estimators=15, max_depth=10, random_state=42)
    rf.fit(X, y)

    xgb = XGBRegressor(n_estimators=15, max_depth=10, random_state=42)
    xgb.fit(X, y)

def ensemble_predict(X):
    return (rf.predict(X) + xgb.predict(X)) / 2

# --- 3. PREDICTION FUNCTION ---
def next_song_prediction(track_name, limit=5, genre_filter=None):
    if df.empty or not track_name:
        return None
        
    track_name = track_name.lower()
    current_song = df[df['track_name'].str.lower() == track_name]

    if current_song.empty:
        return None

    current_song = current_song.iloc[0]

    # Filtering logic - pool candidates to give the model a wide variety to score
    candidates_album = df[
        (df['artist'] == current_song['artist']) & 
        (df['album'] == current_song['album']) & 
        (df['track_name'] != current_song['track_name'])
    ]
    
    candidates_artist = df[
        (df['artist'] == current_song['artist']) & 
        (df['track_name'] != current_song['track_name'])
    ]
    
    candidates_genre = df[
        (df['genre'] == current_song['genre']) & 
        (df['track_name'] != current_song['track_name'])
    ]
    
    candidates_popular = df[df['track_name'] != current_song['track_name']].nlargest(50, 'popularity')

    # Combine candidates and remove duplicates
    candidates = pd.concat([candidates_album, candidates_artist, candidates_genre, candidates_popular])
    candidates = candidates.drop_duplicates(subset=['track_name'])

    # Optional: apply genre filter if provided
    if genre_filter and genre_filter.lower() != 'all':
        genre_filtered = candidates[candidates['genre'].str.lower() == genre_filter.lower()]
        if not genre_filtered.empty:
            candidates = genre_filtered

    if candidates.empty:
        return None

    candidates = candidates.copy()
    candidates['score'] = ensemble_predict(candidates[features])
    
    # Return top N recommendations
    top_songs = candidates.sort_values(by=['score', 'popularity'], ascending=False).head(limit)
    
    # Convert to list of dictionaries
    recommendations = []
    for idx, song in top_songs.iterrows():
        recommendations.append({
            'track_name': song['track_name'],
            'artist': song['artist'],
            'album': song['album'],
            'genre': song['genre'],
            'popularity': float(song['popularity']),
            'score': float(song['score'])
        })
    
    return recommendations