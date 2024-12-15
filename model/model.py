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

    def train(self, x_train, y_train):
        """
        Train the sentiment analysis model
        """
        self.__preprocess()
        self.__fit_naive_bayes(x_train, y_train)
        self.__fit_logistic_regression(x_train, y_train)

    def __fit_naive_bayes(self, x_train, y_train):
        """
        Train Naive Bayes classifier
        """
        self.nb_classifier = MultinomialNB()
        self.nb_classifier.fit(x_train, y_train)

    def __fit_logistic_regression(self, x_train, y_train):
        """
        Train Logistic Regression classifier
        """
        self.lr_classifier = LogisticRegression(max_iter=1000)
        self.lr_classifier.fit(self.X_train, self.y_train)
    
    def __predicit_naive_bayes(self, x_test, y_test):
        """
        Predict using Naive Bayes classifier
        """
        y_pred = self.nb_classifier.predict(x_test)
        print("Naive Bayes")
        print("Accuracy: ", accuracy_score(y_test, y_pred))
        print("Classification Report: ", classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True)
        plt.show()
    
    def __predict_logistic_regression(self, x_test, y_test):
        """
        Predict using Logistic Regression classifier
        """
        y_pred = self.lr_classifier.predict(x_test)
        print("Logistic Regression")
        print("Accuracy: ", accuracy_score(y_test, y_pred))
        print("Classification Report: ", classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True)
        plt.show()
    
    def evaluate(self, x_test, y_test):
        """
        Evaluate the sentiment analysis model
        """
        self.__predicit_naive_bayes(x_test, y_test)
        self.__predict_logistic_regression(x_test, y_test)
        
    
    