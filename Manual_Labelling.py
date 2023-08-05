import pandas as pd
import os
# Load the dataset from CSV file
df = pd.read_csv('D:\CODING\SMT6\AnalisisSentimen_Streamlit\AnalisisSentimenKanjuruhanMalang\Preprocesed+Labelled\Preprocessed_Dataset_09-04-2023_17-19.csv')
df = df.dropna(subset=['tweet'])
# Convert 'tweet' column to string data type
df['tweet'] = df['tweet'].astype(str)
# Create a dictionary to map labels to input values
label_map = {'1': 'positive', '0': 'negative', '2': 'neutral'}

# Create an empty list to store the labels
labels = []

# Loop through each row in the dataset
for index, row in df.iterrows():
    print(f"({index+1}/{len(df)})")
    # Print the text to be labeled
    print(row['tweet'])
    
    # Ask the user to input the label
    label_input = input("Enter label (1=positive, 0=negative, 2=neutral): ")
    
    # Check if the input is valid
    while label_input not in ['0', '1', '2']:
        print("Invalid input. Please enter 0, 1, or 2.")
        label_input = input("Enter label (1=positive, 0=negative, 2=neutral): ")
    
    os.system('cls')
    
    # Map the input value to its corresponding label
    label = label_map[label_input]
    
    # Append the label to the list
    labels.append(label)
    
# Add the labels as a new column to the dataset
df['label'] = labels
# Save the labeled dataset to a new CSV file
df.to_csv('D:\CODING\SMT6\AnalisisSentimen_Streamlit\AnalisisSentimenKanjuruhanMalang\Preprocesed+Labelled\Labelled_Manual_Dataset.csv',encoding='utf-8', index=False)
