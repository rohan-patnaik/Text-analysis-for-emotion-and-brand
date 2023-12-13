from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

def text_processing(tweet):
    
    # Generating the list of words in the tweet (hastags and other punctuations removed)
    def form_sentence(tweet):
        tweet_blob = TextBlob(tweet)
        return ' '.join(tweet_blob.words)
    new_tweet = form_sentence(tweet)
    
    # Removing stopwords and words with unusual symbols
    def no_user_alpha(tweet):
        tweet_list = [ele for ele in tweet.split() if ele != 'user']
        clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
        clean_s = ' '.join(clean_tokens)
        clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]
        return clean_mess
    no_punc_tweet = no_user_alpha(new_tweet)
    
    # Normalizing the words in tweets 
    def normalization(tweet_list):
        lem = WordNetLemmatizer()
        normalized_tweet = []
        for word in tweet_list:
            normalized_text = lem.lemmatize(word,'v')
            normalized_tweet.append(normalized_text)
        return normalized_tweet
    
    
    return normalization(no_punc_tweet)

# Load the sentiment analysis model
with open('NBpipeline.pickle', 'rb') as f:
    loaded_pipe = pickle.load(f)

def predict_sentiment(model, text):
    if not isinstance(text, str):
        raise ValueError("Input should be a string")

    preprocessed_text = text_processing(text)
    predictions = model.predict(preprocessed_text)[0]
    return predictions


@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = ""
    if request.method == 'POST':
        text = request.form['text']
        sentiment = predict_sentiment(loaded_pipe, text)
        # sentiment = text
    return render_template('index.html', sentiment=sentiment)

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    sentiment = predict_sentiment(text)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
