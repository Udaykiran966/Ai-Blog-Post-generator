from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import spacy
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
CORS(app)
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your OpenAI API key as an environment variable

# Function to generate blog posts
def generate_blog_post(topic, style):
    prompt = f"Write a {style} blog post about {topic}."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800
    )
    return response['choices'][0]['message']['content'].strip()

# Function to extract keywords
def extract_keywords(text, num_keywords=10):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    return feature_array[tfidf_sorting][:num_keywords].tolist()

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    topic = data.get("topic")
    style = data.get("style", "formal")
    
    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    
    blog_post = generate_blog_post(topic, style)
    keywords = extract_keywords(blog_post)
    
    return jsonify({"blog_post": blog_post, "keywords": keywords})

if __name__ == "__main__":
    app.run(debug=True)
