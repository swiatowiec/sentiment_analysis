import pandas as pd

df = pd.read_csv('data/Data_tweets.csv', header=None, names=['polarity', 'id', 'date', 'query', 'user', 'text'])

# PREPROCESSING
def remove_punctuation(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":",  "!",'"', "-"))
    return final
df['text'] = df['text'].apply(remove_punctuation)
df = df[['text','polarity']]
