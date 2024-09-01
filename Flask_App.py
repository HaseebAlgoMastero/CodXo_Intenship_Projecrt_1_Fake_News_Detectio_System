# Web application for the News reliability checker using the flask App 

from flask import Flask, render_template_string, request
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

nltk.download("stopwords")
ps = PorterStemmer()
Stopwords = set(stopwords.words('english'))

vector_form = pickle.load(open('vector.pkl', 'rb'))
classifier_LG = pickle.load(open('classifier.pkl', 'rb'))

def lower_text(text):
    return text.lower()
def punctuation_removal(text):
    punctuations = string.punctuation
    return text.translate(str.maketrans('', '', punctuations))
def removal_of_words(text):
    return " ".join([word for word in text.split() if word not in Stopwords])
def stem_words(text):
    return " ".join([ps.stem(word) for word in text.split()])
def clean_text(text):
    text = re.sub(r'\[.*?\]|https?://\S+|www\.\S+|<.*?>+|\n|\w*\d\w*', '', text)
    return text
app = Flask(__name__)

# Homepage with an interactive header and additional content
@app.route('/')
def index():
    return render_template_string('''
    <html>
    <head>
        <title>Fake News Detection</title>
        <style>
           body {
           font-family: 'Poppins', sans-serif;
           background-color: #121212;
           color: #e0e0e0;
           margin: 0;
           padding: 0;
           }
           header {
           text-align: center;
           padding: 40px 20px;
           background-color: #1f1f1f;
           border-bottom: 2px solid #333;
           }
           header h1 {
           font-size: 3rem;
           font-weight: 600;
           margin: 0;
           color: #ffffff;
           }
           header p {
           font-size: 1.2rem;
           font-weight: 300;
           color: #b0b0b0;
           margin-top: 10px;
           }
           section {
           max-width: 800px;
           margin: 40px auto;
           padding: 20px;
           background-color: #1c1c1c;
           border-radius: 10px;
           box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
           }
           section h2 {
           font-size: 2rem;
           font-weight: 400;
           margin-bottom: 15px;
           color: #ffffff;
           }
           section p {
           font-size: 1rem;
           line-height: 1.6;
           color: #d0d0d0;
           }
           textarea {
           width: 80%;
           height: 180px;
           padding: 15px;
           margin-bottom: 20px;
           border: none;
           border-radius: 8px;
           background-color: #2a2a2a;
           color: #f0f0f0;
           font-size: 1rem;
           font-family: 'Poppins', sans-serif;
           resize: none;
           }
           textarea::placeholder {
           color: #888;
           }
           button {
           background-color: #5c67f2;
           color: #ffffff;
           padding: 12px 25px;
           font-size: 1rem;
           font-weight: 500;
           border: none;
           border-radius: 8px;
           cursor: pointer;
           transition: transform 0.2s, background-color 0.3s;
           }
           button:hover {
           background-color: #4b56e0;
           transform: translateY(-3px);
           }
           footer {
           text-align: center;
           padding: 20px;
           background-color: #1a1a1a;
           color: #999;
           font-size: 0.9rem;
           margin-top: 20px;
           }
        </style>
    </head>
    <body>
       <header>
    <h1>Fake News Classifier</h1>
    <p>Detecting fake news using AI-powered analysis</p>
  </header>

  <section class="about">
    <h2>About the Project</h2>
    <p>This project is designed to classify news articles as fake or real using advanced machine learning models. Paste a news article below and let our AI analyze it for you.</p>
     <h4>Coded By: Haseeb Iqbal</h4>
     
    <h4>Supported By the CodXo</h4>
    
    
  </section>

  <section class="input-section">
    <h2>Input News Article</h2>
    <form action="/predict" method="post">
        <textarea name="news_text" id="newsInput" placeholder="Enter or paste the news article text here..."></textarea>
        <button type="submit" id="submitBtn">Classify News</button>
    </form>
  </section>

  <footer>
    <p>© 2024 Fake News Classifier. All rights reserved.</p>
  </footer>
    </body>
    </html>
    ''')


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news_text']

    # Preprocess the input text
    news = lower_text(news_text)
    news = punctuation_removal(news)
    news = removal_of_words(news)
    news = stem_words(news)
    news = clean_text(news)

    # Transform the input text
    vector_form1 = vector_form.transform([news])

    # Make a prediction
    prediction = classifier_LG.predict(vector_form1)[0]
    label = "Fake News" if prediction == 0 else "Real News"

    # Render the result page
    return render_template_string(f'''
    <html>
    <head>
        <title>Fake News Detection - Result</title>
        <style>
            body {{
                background-color: #121212;
                font-family: 'Arial', sans-serif;
                color: #f4f4f4;
                text-align: center;
                padding: 50px;
            }}
            h1 {{
                font-size: 2.5em;
                font-weight: bold;
                color: #ff6b6b;
                margin-bottom: 20px;
            }}
            .result {{
                font-size: 2em;
                padding: 20px;
                background-color: {'#ff6b6b' if prediction == 0 else '#2ecc71'};
                color: #fff;
                display: inline-block;
                border-radius: 10px;
                margin-bottom: 20px;
            }}
            a {{
                display: block;
                margin-top: 20px;
                font-size: 1.2em;
                color: #ff6b6b;
                text-decoration: none;
            }}
            a:hover {{
                color: #ff4747;
            }}
        </style>
    </head>
    <body>
        <h1>Prediction Result</h1>
        <div class="result">{label}</div>
        <a href="/">Try Again</a>
    </body>
    </html>
    ''')

if __name__ == "__main__":
    app.run(debug=True)
