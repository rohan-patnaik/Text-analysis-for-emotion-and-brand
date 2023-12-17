from flask import Flask, render_template, request
from transformers import AutoTokenizer
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Path to your fine-tuned RoBERTa model
MODEL_PATH = './models/roberta_emotion_classifier'

# Load tokenizer
TOKENIZER_NAME = 'cardiffnlp/twitter-roberta-base-emotion-multilabel-latest'
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# Load your custom RoBERTa model
model = tf.keras.models.load_model(MODEL_PATH)

# Function to get sentiment using your custom RoBERTa model
def polarity_scores_roberta(text):
    try:
        encoded_text = tokenizer(text, return_tensors='tf', truncation=True, max_length=512)
        output = model(encoded_text)
        scores = output.numpy()
        label_mapping = ['Negative', 'Neutral', 'Positive']  # Update this if needed
        max_label = label_mapping[np.argmax(scores)]
        return max_label
    except IndexError as e:
        print("An error occurred with label mapping:", e)
        raise IndexError("The index for the maximum score is out of the label mapping range.")

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
        sentiment = polarity_scores_roberta(inp)
        brand = brand_mapper(inp)
        return render_template('home.html', message=sentiment, brand=brand)
    return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
