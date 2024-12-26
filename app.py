import joblib
from flask import Flask, request, jsonify, render_template
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load("modelrev.pkl")
vec = joblib.load("vectorizer.pkl")

# Initialize NLTK utilities
lemmatizer = WordNetLemmatizer()
en_stopwords = stopwords.words('english')

# Function for prediction
def prediction(sentence):
    labels = ["negative", "positive"]
    sentence = sentence.lower()
    punct = re.sub(r"([^\w\s])", "", sentence)
    no_stop_words = " ".join(
        [word for word in punct.split() if word not in en_stopwords]
    )
    tokenizer = word_tokenize(no_stop_words)
    bag = [" ".join(tokenizer)]
    feature = vec.transform(bag)
    prediction = model.predict(feature)
    return labels[prediction[0]]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data['text']
        result = prediction(text)
        return jsonify({'status': 'success', 'prediction': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
