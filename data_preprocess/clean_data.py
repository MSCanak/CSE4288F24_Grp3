import pandas as pd
import re

# Load the dataset
data = pd.read_csv("./data/IMDB _Dataset.csv")

# Remove duplicates
data = data.drop_duplicates(subset="review", keep="first")

# Remove HTML tags
data['review'] = data['review'].str.replace(r'<.*?>', '', regex=True)

# Convert text to lowercase
data['review'] = data['review'].str.lower()

# Remove punctuation and special characters
data['review'] = data['review'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

# Save the cleaned dataset
data.to_csv("./data/cleaned_IMDB_dataset.csv", index=False)
