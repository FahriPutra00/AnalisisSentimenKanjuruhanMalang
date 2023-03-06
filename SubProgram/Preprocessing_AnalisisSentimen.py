import nltk
import os
import glob
import re
import string
import numpy as np
import pandas as pd
import re

# import library unutk stopword
from nltk.corpus import stopwords
stop_words = stopwords.words('indonesian')

# import library untuk stemming
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# import library untuk tokenisasi
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer

#list all csv files only
csv_files = glob.glob(pathname='*.{}'.format('csv'), root_dir='D:\CODING\SMT5\ICP\DATASET_06-10-2022_19-29')

dataSet = pd.concat([pd.read_csv(filepath_or_buffer='D:\CODING\SMT5\ICP\DATASET_06-10-2022_19-29\{}'.format(f)) for f in csv_files ], ignore_index=True)

# Membuang kolom tanggal
dataSet.drop(['user_name'], axis=1, inplace=True)
dataSet.drop(['Unnamed: 0'], axis=1, inplace=True)
dataSet.drop_duplicates(subset ="teks", keep = 'first', inplace = True)
dataSet.dropna()

def text_prep_norm(teks):
    # remove old style retweet teks "RT"
    teks = re.sub(r'^RT[\s]+', ' ', teks)
    
    # convert to lowercase
    teks = teks.lower()
    
    # remove numbers
    teks = re.sub(r'\d+', '', teks)
    
    # remove stock market tickers like $GE
    teks = re.sub(r'\$\w*', '', teks)
            
    # remove hashtags
    # only removing the hash # sign from the word
    teks = re.sub(r'#', '', teks)
    
    # remove @ mentions with underscore
    teks = re.sub(r'@\w*_[A-Za-z0-9_]*', '', teks)
    
    # remove @ mentions without underscore
    teks = re.sub(r'@\w*', '', teks)

    # remove sym
    teks = re.sub(r":|;|=|-|\)|\(|\*|'", '', teks)
        
    # remove links starting with "http" or "https"
    teks = re.sub(r'http\S+', '', teks)
    teks = re.sub(r'https\S+', '', teks)

    # remove "tco" mentions
    teks = re.sub(r'tco/\S+', '', teks)

    # remove punctuation marks and symbols
    teks = re.sub(r'[^\w\s]', '', teks)
    
    # remove underscore
    teks = re.sub(r'_', '', teks)

    # remove extra whitespace
    teks = re.sub(r'\s+', ' ', teks)
    
    teks = teks.strip()

    return teks

dataSet['tweet_regex'] = [text_prep_norm(i) for i in dataSet['teks']]

def normalize_text(teks):
    # Load slang word datasets
    slang_datasets = [
        'https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/master/new_kamusalay.csv',
        'https://raw.githubusercontent.com/panggi/pujangga/master/resource/formalization/formalizationDict.txt',
        'https://raw.githubusercontent.com/agusmakmun/SentiStrengthID/master/id_dict/slangword.txt'
    ]
    slang_words = pd.concat([pd.read_csv(slang, sep='\t|:', header=None, names=['slang', 'normal'], engine='python', encoding='unicode_escape') for slang in slang_datasets], ignore_index=True)
    
    # Add more slang words to the dataset
    more_slang = pd.DataFrame({
        'slang': ['sblm', 'sdh', 'laah', 'jlas', 'mnjlaskn', 'bhw', 'mmc', 'islamkaffah', 'serbaserbi', 'amp'], 
        'normal': ['sebelum', 'sudah', 'lah', 'jelas', 'menjelaskan', 'bahwa', '', '', '', '']
    })
    slang_words = pd.concat([slang_words, more_slang], ignore_index=True)
    
    # Remove duplicates and convert to dictionary
    slang_dict = dict(slang_words.drop_duplicates(subset='slang', keep='first').values)
    
    # Replace slang words
    teks = ' '.join([slang_dict.get(word, word) for word in teks.split()])
    
    # Remove all non-alphanumeric characters and extra spaces
    teks = re.sub(r'[^\w\s]+', '', teks).strip()    
    return teks

dataSet['tweet_normal'] = dataSet['tweet_regex'].apply(normalize_text)

lemma = WordNetLemmatizer()

# Happy Emoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])
 
# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])
 
# all emoticons (happy + sad)
emoticons = emoticons_happy.union(emoticons_sad)
def text_lemmatize(teks):
    tokens = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True).tokenize(teks)
    kalimat = [t for t in tokens if t not in stop_words and t not in emoticons and t not in string.punctuation and len(t) > 3]
    lemmatize = [lemma.lemmatize(w) for w in kalimat]

    return lemmatize

dataSet['tweet_lemmatize'] = [text_lemmatize(i) for i in dataSet['tweet_normal']]

#remove punct
def remove_punct(text):
    text  = " ".join([char for char in text if char not in string.punctuation])
    return text
dataSet['tweet'] = dataSet['tweet_lemmatize'].apply(lambda x: remove_punct(x))

dataSet.drop(dataSet.columns[[0,1]], axis = 1, inplace = True)
dataSet.drop_duplicates(subset ="tweet", keep = 'first', inplace = True)
dataSet.to_csv('Kanjuruhan_TEST_NORMALISASI.csv',encoding='utf-8', index=False)

