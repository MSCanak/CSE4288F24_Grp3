import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re

# Load the dataset
file_path = '/content/IMDB Dataset.csv'
data = pd.read_csv(file_path)

# handle duplicates
print(f"Original dataset size: {data.shape}")
data = data.drop_duplicates(subset='review', keep='first')
print(f"Dataset size after removing duplicates: {data.shape}")

# handle html tags
data['review'] = data['review'].str.replace(r'<.*?>', '', regex=True)
print(data['review'])

# normalize case
data['review'] = data['review'].str.replace(r'<.*?>', '', regex=True)
print(data['review'])

# Remove Punctuation and Special Characters
data['review'] = data['review'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
print(data['review'])

# handle missing values
print(f"Missing values:\n{data.isnull().sum()}")

# sentiment distribution
sentiment_counts = data['sentiment'].value_counts()
print(f"Sentiment Distribution:\n{sentiment_counts}")
sentiment_counts.plot(kind='bar', title='Sentiment Distribution', color=['orange', 'blue'])
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# review length statistics
data['review_length'] = data['review'].apply(len)
print(f"Review Length Statistics:\n{data['review_length'].describe()}")

# review length distribution
plt.hist(data['review_length'], bins=50, alpha=0.7, color='green')
plt.title('Review Length Distribution')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
plt.show()

# word count statistics
data['word_count'] = data['review'].apply(lambda x: len(x.split()))
print(f"Word Count Statistics:\n{data['word_count'].describe()}")

# word count distribution
plt.hist(data['word_count'], bins=50, alpha=0.7, color='purple')
plt.title('Word Count Distribution')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.show()

# most common words
all_words = ' '.join(data['review']).split()
common_words = Counter(all_words).most_common(10)
print(f"Most Common Words:\n{common_words}")

# word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(data['review']))
plt.figure(figsize=(10, 5))
plt.title("Word Cloud for All Reviews")
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# sentiment based word frequency
positive_reviews = ' '.join(data[data['sentiment'] == 'positive']['review'])
negative_reviews = ' '.join(data[data['sentiment'] == 'negative']['review'])

# positive sentiment
pos_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)
plt.figure(figsize=(10, 5))
plt.title("Positive Reviews WordCloud")
plt.imshow(pos_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# most common words in positive reviews
positive_words = positive_reviews.split()
positive_common_words = Counter(positive_words).most_common(10)
print(f"Most Common Words in Positive Reviews:\n{positive_common_words}")

# negative sentiment
neg_wordcloud = WordCloud(width=800, height=400, background_color='black').generate(negative_reviews)
plt.figure(figsize=(10, 5))
plt.title("Negative Reviews WordCloud")
plt.imshow(neg_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# most common words in negative reviews
negative_words = negative_reviews.split()
negative_common_words = Counter(negative_words).most_common(10)
print(f"Most Common Words in Negative Reviews:\n{negative_common_words}")

# sentiment encoding
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000)  # Limit to 5000 most important words
X = tfidf.fit_transform(data['review'])
y = data['sentiment']

print(f"TF-IDF matrix shape: {X.shape}")

# Save the preprocessed dataset
data.to_csv('cleaned_IMDB_dataset.csv', index=False)
print("Preprocessed dataset saved as 'cleaned_IMDB_dataset.csv'")