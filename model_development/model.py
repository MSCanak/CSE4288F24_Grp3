import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


class MovieReviewSentimentAnalysis:
    def __init__(self, dataset_path):
        self.dataset = pd.read_csv(dataset_path)
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

    def __preprocess(self):
        x = self.vectorizer.fit_transform(self.dataset["review"])
        y = self.dataset["sentiment"]
        return train_test_split(x, y, test_size=0.2, random_state=42)

    def train(self):
        X_train, X_test, y_train, y_test = self.__preprocess()
        self.__fit_naive_bayes(X_train, y_train)
        self.__fit_logistic_regression(X_train, y_train)
        self.X_test = X_test  # Save test data for evaluation
        self.y_test = y_test

    def __fit_naive_bayes(self, X_train, y_train):
        self.nb_classifier = MultinomialNB()
        self.nb_classifier.fit(X_train, y_train)

    def __fit_logistic_regression(self, X_train, y_train):
        self.lr_classifier = LogisticRegression(max_iter=1000)
        self.lr_classifier.fit(X_train, y_train)

    def __predict_naive_bayes(self):
        y_pred = self.nb_classifier.predict(self.X_test)
        print("Naive Bayes")
        print("Accuracy: ", accuracy_score(self.y_test, y_pred))
        print("Classification Report: ", classification_report(self.y_test, y_pred))
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.show()

    def __predict_logistic_regression(self):
        y_pred = self.lr_classifier.predict(self.X_test)
        print("Logistic Regression")
        print("Accuracy: ", accuracy_score(self.y_test, y_pred))
        print("Classification Report: ", classification_report(self.y_test, y_pred))
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.show()

    def evaluate(self):
        self.__predict_naive_bayes()
        self.__predict_logistic_regression()


a = MovieReviewSentimentAnalysis("../data/cleaned_IMDB_dataset.csv")
a.train()
a.evaluate()
