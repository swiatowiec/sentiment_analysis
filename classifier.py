import pandas as pd

df = pd.read_csv('data/Data_tweets.csv', header=None, names=['polarity', 'id', 'date', 'query', 'user', 'text'])
print(df.head())