import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


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
        self.lr_classifier = LogisticRegression(max_iter=1000)
        self.lr_classifier.fit(X_train, y_train)
        self.X_test = X_test  # Save test data for evaluation
        self.y_test = y_test

    def evaluate(self):
        y_pred = self.lr_classifier.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)

        # Print results to console
        print("Logistic Regression")
        print("Accuracy: ", accuracy)
        print("Classification Report: \n", report)

        # Save results to a file
        with open("output_results.txt", "w") as f:
            f.write("Logistic Regression\n")
            f.write(f"Accuracy: {accuracy}\n\n")
            f.write("Classification Report:\n")
            f.write(report + "\n")

        # Plot and save confusion matrix
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix - Logistic Regression")
        plt.savefig("output_confusion_matrix.png")
        plt.show()


# Kullanıcıdan CSV dosyasının yolunu al
dataset_path = input("Lütfen CSV dosyasının yolunu girin: ")
a = MovieReviewSentimentAnalysis(dataset_path)
a.train()
a.evaluate()