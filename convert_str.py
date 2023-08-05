import pandas as pd
import os

# Load the dataset from CSV file
df = pd.read_csv('D:\CODING\SMT6\AnalisisSentimen_Streamlit\AnalisisSentimenKanjuruhanMalang\Preprocesed+Labelled\Labelled_Manual_Dataset.csv')

# Drop rows with NaN values in the 'tweet' column
df = df.dropna(subset=['tweet'])

# Convert 'tweet' column to string data type
df['tweet'] = df['tweet'].astype(str)

# Save the fixed dataset to a new CSV file
df.to_csv('D:\CODING\SMT6\AnalisisSentimen_Streamlit\AnalisisSentimenKanjuruhanMalang\Preprocesed+Labelled\Labelled_Manual_Dataset_Fix.csv', encoding='utf-8', index=False)
