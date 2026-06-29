from flask import Flask, request, jsonify, render_template
from spotify_recommendation import next_song_prediction, df
import pandas as pd

# Initialize Flask (Defaults: template_folder='templates', static_folder='static')
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    next_song = None
    if request.method == "POST":
        # Get inputs from form
        song_name = request.form.get("track_name")
        genre_filter = request.form.get("genre_filter", "all")
        try:
            limit = int(request.form.get("limit", 5))
        except ValueError:
            limit = 5
        
        # Get prediction
        next_song = next_song_prediction(song_name, limit=limit, genre_filter=genre_filter)
    
    # get distinct genres for the dropdown
    genres = []
    if not df.empty:
        genres = sorted(df['genre'].dropna().unique().tolist())
        
    return render_template("index.html", next_song=next_song, genres=genres)

@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "").lower()
    if not query or df.empty:
        return jsonify([])
    
    matches = df[df['track_name'].str.lower().str.contains(query, na=False)]
    results = matches['track_name'].drop_duplicates().head(10).tolist()
    return jsonify(results)

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json
    song = data.get("song")
    result = next_song_prediction(song)
    if result is None:
        return jsonify({"error": "Song not found"})
    # result is now a list of recommendations
    return jsonify({"recommendations": result})

@app.route("/songs", methods=["GET"])
def get_all_songs():
    """Get all songs from dataset for playlist"""
    if df.empty:
        return jsonify([])
    
    songs_list = df[['track_name', 'artist', 'album', 'genre', 'popularity']].drop_duplicates().to_dict('records')
    return jsonify(songs_list)

import os

if __name__ == "__main__":
    # Render assigns a dynamic port via the PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)