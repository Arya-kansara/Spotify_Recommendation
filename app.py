from flask import Flask, request, jsonify, render_template
from spotify_recommendation import next_song_prediction
import os

# 1. TELL FLASK TO LOOK IN THE CURRENT FOLDER FOR HTML & CSS
app = Flask(__name__, template_folder='.', static_folder='.')

@app.route("/", methods=["GET", "POST"])
def home():
    next_song = None
    
    # 2. IF USER SUBMITS THE FORM
    if request.method == "POST":
        # Get the song name entered in the input box
        song_name = request.form.get("track_name")
        
        # Run the ML model
        next_song = next_song_prediction(song_name)
    
    # 3. SHOW THE PAGE (PASSING THE RESULT IF WE HAVE ONE)
    return render_template("index.html", next_song=next_song)

# (Optional) Keep this separate API endpoint if you still want to use Postman
@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json
    song = data.get("song")
    result = next_song_prediction(song)
    if result is None:
        return jsonify({"error": "Song not found"})
    return jsonify(result)

if __name__ == "__main__":
    # Use port 10000 for Render
    app.run(host="127.0.0.1", port=5000)