# Text analyzer for emotion and brand

Start by opening and reading the Approach.pdf file in thie repo to understand my appraoch and how I took on the questions asked my the assignment.
There are 4 notebooks here for you to go through. wysa.ipynb and wysa2.ipynb, Custom_Emotion_Analysis.ipynb and Custom_Emotion_Analysis_Roberta.ipynb.

The Custom_Emotion_Analysis.ipynb has A-Z for analysis of emotion in our given data file. Consists of imbalanced data handling, data augmentation, finetuning a pretrained transformer model and makes predictions on given text!
This notebook makes use of bert-base-uncased model in my approach.

(the models folder for saved models was too large to upload so you may find the drive link for it here :https://drive.google.com/drive/folders/1jsX528JA9g3eEt4d6jx5uCvTfFredWMx?usp=drive_link)

wysa.ipynb and wysa2.ipynb are still a bit rough around the edges(I'll improve it).

wysa.ipynb has full EDA, emotion, tweets and brand/product analysis. It also has a detailed plan on how to structure a pipeline if the given task is to be achieved as a full fledged application.\
And lastly it has also displays the use of RandomForest approach and VADER approach for sentiment classification.

wysa2.ipynb has NaiveBayes approach, Roberta approach and a complete brand/product identification using custom NER on top of BERT(the code is fine however its failing coz of some resource issue which i will fix)
Updated the "cardiffnlp/twitter-roberta-base-sentiment" model with "cardiffnlp/twitter-roberta-base-sentiment-latest" for better sarcasm and irony detection.

You can find the brand prediction for test data in the file submission_test_data.csv

To run the web-app after cloning run the command 'python app.py'

![image](https://github.com/rohan-patnaik/Text-analysis-for-emotion-and-brand/assets/22250758/c8d237c7-c79d-4cc7-90e1-b5fcb2ef1f33)


