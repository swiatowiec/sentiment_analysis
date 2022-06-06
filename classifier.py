import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('data/Data_tweets.csv', header=None, names=['polarity', 'id', 'date', 'query', 'user', 'text'])
TRAIN_TEST_PROP = 0.8

# PREPROCESSING
def remove_punctuation(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":",  "!",'"', "-"))
    return final
df['text'] = df['text'].apply(remove_punctuation)
df = df[['text','polarity']]

# MODELING
index = df.index
df['random_number'] = np.random.randn(len(index))
train = df[df['random_number'] <= TRAIN_TEST_PROP]
test = df[df['random_number'] > TRAIN_TEST_PROP]

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train['text'])
test_matrix = vectorizer.transform(test['text'])

lr = LogisticRegression()

X_train = train_matrix
X_test = test_matrix
y_train = train['polarity']
y_test = test['polarity']

lr.fit(X_train,y_train)