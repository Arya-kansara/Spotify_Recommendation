# Spotify Music Recommendation System 🎵

A Machine Learning-based web application that suggests songs based on your musical preferences. Built using an ensemble of **Random Forest** and **XGBoost** models, this app dynamically filters and ranks songs from a massive dataset to provide you with the best matches.

## Features ✨
- **Intelligent Recommendations**: Combines candidate pooling (artist, album, genre) with machine learning scoring to guarantee high-quality recommendations.
- **Search Autocomplete**: Instantly suggests songs as you type, so you never have to worry about typos or exact names.
- **Dynamic Filtering**: Filter your recommendations by specific genres or choose exactly how many songs you want (5, 10, or 20).
- **Premium User Interface**: Features a stunning, fully responsive dark-mode UI with glassmorphism effects, smooth animations, and the sleek "Outfit" font.
- **Ready for Production**: Pre-configured for seamless deployment to Render using Gunicorn.

## Tech Stack 🛠️
- **Backend Framework**: Python / Flask
- **Machine Learning**: Scikit-Learn (Random Forest, LabelEncoder), XGBoost, Pandas
- **Frontend**: HTML5, CSS3 (Vanilla, No Frameworks)
- **Deployment Server**: Gunicorn

## How to Run Locally 💻

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd Spotify_Recommendation
   ```

2. **Install the dependencies:**
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open in Browser:**
   Visit `http://localhost:10000` (or whatever port is shown in your terminal) to view the app!

## Deployment 🚀 (Render)

This project is pre-configured for automated deployment on [Render](https://render.com) using Infrastructure as Code (`render.yaml`).

1. Push your code to a GitHub repository.
2. Log into Render and click **New +** -> **Blueprint**.
3. Connect your GitHub repository.
4. Render will automatically detect the `render.yaml` file, install the requirements, and deploy your app using the `gunicorn app:app` command.

## Dataset
Ensure the `spotify_dataset.csv` file is in the root directory for the application to function correctly. The dataset contains metadata such as `track_name`, `artist`, `album`, `genre`, and `popularity` which are used for filtering and ML predictions.
