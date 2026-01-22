from flask import Flask, request, jsonify, render_template
from spotify_recommendation import next_song_prediction

# Initialize Flask (Defaults: template_folder='templates', static_folder='static')
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    next_song = None
    if request.method == "POST":
        # Get song name from form
        song_name = request.form.get("track_name")
        # Get prediction
        next_song = next_song_prediction(song_name)
    
    return render_template("index.html", next_song=next_song)

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json
    song = data.get("song")
    result = next_song_prediction(song)
    if result is None:
        return jsonify({"error": "Song not found"})
    return jsonify(result)

if __name__ == "__main__":
    # Port 10000 is required for Render
    app.run(host="0.0.0.0", port=10000)
