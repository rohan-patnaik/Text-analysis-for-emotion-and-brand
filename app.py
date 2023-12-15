from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np

app = Flask(__name__)

# RoBERTa model setup
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Function to get sentiment using RoBERTa
def polarity_scores_roberta(text):
    encoded_text = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    label_mapping = ['Negative', 'Neutral', 'Positive']
    max_label = label_mapping[np.argmax(scores)]
    
    return max_label

def brand_mapper(label):
    normalized_label = label.lower().replace(" ", "")

    apple_keywords = ['ipad', 'iphone', 'apple']
    google_keywords = ['google', 'android']

    if any(keyword in normalized_label for keyword in apple_keywords):
        return 'Apple'
    elif any(keyword in normalized_label for keyword in google_keywords):
        return 'Google'
    else:
        return 'No brand identified'

@app.route('/', methods=["GET", "POST"])
def main():
    if request.method == "POST":
        inp = request.form.get("inp")
        
        # Use RoBERTa model for sentiment analysis
        sentiment = polarity_scores_roberta(inp)

        brand = brand_mapper(inp)

        return render_template('home.html', message=sentiment, brand=brand)
    return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
