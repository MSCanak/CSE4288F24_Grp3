import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

class MovieReviewSentimentAnalysis:
    def __init__(self, dataset_path):
        """
        Initialize the sentiment analysis model
        
        :param dataset_path: Path to the IMDB movie reviews dataset
        """
        self.dataset = pd.read_csv(dataset_path)

    def train_naive_bayes(self):
        """
        Train Naive Bayes classifier
        """
        self.nb_classifier = MultinomialNB()
        self.nb_classifier.fit(self.X_train, self.y_train)
        
        # Predictions
        nb_predictions = self.nb_classifier.predict(self.X_test)
        
        print("Naive Bayes Classifier Performance:")
        print(classification_report(self.y_test, nb_predictions))
        
        return nb_predictions
    
    