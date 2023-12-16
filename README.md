# Text analyzer for emotion and brand

Start by opening and reading the Approach.pdf file in thie repo to understand my appraoch and how I took on the questions asked my the assignment.
There are 2 notebooks here for you to go through. wysa.ipynb and wysa2.ipynb

wysa.ipynb has full EDA, emotion, tweets and brand/product analysis. It also has a detailed plan on how to structure a pipeline if the given task is to be achieved as a full fledged application.\
And lastly it has also displays the use of RandomForest approach and VADER approach for sentiment classification.

wysa2.ipynb has NaiveBayes approach, Roberta approach and a complete brand/product identification using custom NER on top of BERT(the code is fine however its failing coz of some resource issue which i will fix)
Updated the "cardiffnlp/twitter-roberta-base-sentiment" model with "cardiffnlp/twitter-roberta-base-sentiment-latest" for better sarcasm and irony detection.

I am trying to improve/tinker with the code. I won't break the code(hopefully) since I am trying to add a new method to analyse the brand being targetted in any specific tweet.
You can find the brand prediction for test data in the file submission_test_data.csv

To run the web-app after cloning run the command 'python app.py'

![image](https://github.com/rohan-patnaik/Text-analysis-for-emotion-and-brand/assets/22250758/c8d237c7-c79d-4cc7-90e1-b5fcb2ef1f33)


