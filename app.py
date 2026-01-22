from flask import Flask, request, jsonify
from spotify_recommendation import next_song_prediction

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Spotify ML Recommendation API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    song = data.get("song")
    result = next_song_prediction(song)

    if result is None:
        return jsonify({"error": "Song not found"})
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
