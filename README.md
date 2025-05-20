from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie data
movies = [
    {"title": "Inception", "genre": "sci-fi thriller dream"},
    {"title": "Interstellar", "genre": "space sci-fi drama"},
    {"title": "The Dark Knight", "genre": "action crime drama"},
    {"title": "Gravity", "genre": "space sci-fi thriller"},
]

# User preference
user_input = "sci-fi space"

# Vectorize genres
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([m['genre'] for m in movies] + [user_input])

# Compute similarity
similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

# Recommend top movie
index = similarity.argmax()
print("Recommended Movie:", movies[index]['title'])
