from flask import Flask, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

app = Flask(__name__)

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
        sia = SentimentIntensityAnalyzer()
        score = sia.polarity_scores(inp)
        if score["neg"] > score["pos"]:
            message = "Negative"
        else:
            message = "Positive"

        brand = brand_mapper(inp)

        return render_template('home.html', message=message, brand=brand)
    return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
