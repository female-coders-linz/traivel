gifrom flask import Flask, render_template, request

# Your code to calculate embeddings and find the best match
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

app = Flask(__name__, static_folder='static')


@app.route('/')
def index():
    return render_template('traiwell.html')

@app.route('/search', methods=['POST'])
def search():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    activity_descriptions = [
        "Explore the historic Old Town of Linz with home of Johan Kepler.",
        "Explore the heart of Austria's capital. Visit the Hofburg Palace, St. Stephen's Cathedral, and more.",
        "Discover the magnificent Schönbrunn Palace, a UNESCO World Heritage site with beautiful gardens.",
        "Enjoy a scenic cruise along the Danube River, taking in the city's stunning skyline.",
        "Visit the Belvedere Palace and its world-renowned art collection.",
        "Hike through the Vienna Woods, a beautiful natural area just outside the city.",
        "Explore the historic Old Town of Salzburg, the birthplace of Mozart.",
        "Visit the iconic fortress perched on a hill with panoramic views of Salzburg.",
        "Follow in the footsteps of the Von Trapp family and visit film locations.",
        "Stroll through the beautiful Mirabell Gardens, featured in the movie 'The Sound of Music.'",
        "Take a tour of the salt mines in the nearby Dürrnberg region.",
        "Discover the historic streets and landmarks of Linz, a vibrant Austrian city.",
        "Explore the futuristic world of digital arts at the Ars Electronica Center.",
        "Visit the historic Linz Castle and learn about its rich history.",
        "Experience contemporary art at the Lentos Art Museum, located on the Danube River.",
        "Take a ride on the Pöstlingbergbahn, a historic mountain tramway with panoramic views.",
        "Hike through the Linz Woods, a beautiful natural area just outside the city."
    ]
    embeddings = model.encode(activity_descriptions)
    query = request.form['query']
    queryEmbedding = model.encode(query)
    cos_sim = []

    for embedding in embeddings:
        cos_sim.append(util.pytorch_cos_sim(queryEmbedding, embedding)[0][0].item())

    best_match_index = np.argmax(cos_sim)
    best_match_description = activity_descriptions[best_match_index]

    return render_template('traiwell.html', query=query, best_match=best_match_description)

if __name__ == '__main__':
    app.run(debug=True)
