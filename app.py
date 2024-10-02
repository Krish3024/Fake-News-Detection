from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Initialize the Porter Stemmer
port_stem = PorterStemmer()

# Function for stemming
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Home route
@app.route('/predict', methods=['POST'])
def predict():
    # Change this part to get data from JSON
    data = request.get_json()  # Get JSON data
    title = data['title']       # Extract title
    author = data['author']     # Extract author
    content = f"{author} {title}"
    
    processed_content = stemming(content)
    
    # Transform the input content
    input_vector = vectorizer.transform([processed_content])
    
    # Make prediction
    prediction = model.predict(input_vector)[0]

    # Map prediction to label
    result = "Real News" if prediction == 0 else "Fake News"
    
    # Return JSON response
    return jsonify({"prediction": result})  # Send result as JSON

if __name__ == '__main__':
    app.run(debug=True)
