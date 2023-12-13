from flask import Flask, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

app = Flask(__name__)  # Corrected the name declaration

@app.route('/', methods=["GET", "POST"])
def main():
    if request.method == "POST":
        inp = request.form.get("inp")
        sia = SentimentIntensityAnalyzer()
        score = sia.polarity_scores(inp)
        if score["neg"] > score["pos"]:
            message = "Negative"  # Added variable assignment
        else:
            message = "Positive"  # Added variable assignment
        return render_template('home.html', message=message)
    return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)  # Added the conditional to run the Flask application
