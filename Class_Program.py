import os
import streamlit as st
import tweepy
import pandas as pd
import datetime as dt
import warnings
from dotenv import load_dotenv
warnings.filterwarnings("ignore")
st.config.set_option("deprecation.showPyplotGlobalUse", False)
@st.cache_data()
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun, index = False to prevent saving the index
    return df.to_csv(sep=',',index=False).encode('utf-8')
            
@st.cache_resource()
class TwitterCrawler():
    load_dotenv()
    
    def __init__(self):
        self.consumer_key = os.getenv('consumer_key')
        self.consumer_secret = os.getenv('consumer_secret')
        self.access_token = os.getenv('access_token')
        self.access_token_secret = os.getenv('access_token_secret')
        self.auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
        self.auth.set_access_token(self.access_token, self.access_token_secret)
        self.api = tweepy.API(self.auth,wait_on_rate_limit=True)
        self.date = dt.datetime.now().strftime("%d-%m-%Y_%H-%M")
    
    def search_tweet(self, keyword, length):
        with st.spinner('Running Crawling Data...'):
            tweets_list = tweepy.Cursor(self.api.search_tweets, q=keyword, tweet_mode='extended', lang='id').items(length)
            output = [{'user_name': tweet.user.screen_name, 'teks': tweet._json['full_text']} for tweet in tweets_list]
        st.success('Crawling Selesai')
        st.write("Berhasil Melakukan Crawling Data Sejumlah :",len(output))    
        filename = f"{keyword}_{len(output)}_{self.date}.csv"
        Dataset = pd.DataFrame(output)
        st.title("Crawling Dataset")
        st.dataframe(Dataset,width=None, height=None, use_container_width=True)
        csv = convert_df(Dataset)
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.empty()
            with col2:
                st.download_button(
                    label="Download Dataset CSV:arrow_down:",
                    data=csv,
                    file_name=filename,
                    mime='text/csv',
                )
            with col3:
                st.empty()

    
    def run(self):
        # censor some words in the token and secret
        consumer_key_censored = self.consumer_key[:4] + "*" * 10 + self.consumer_key[-4:]
        consumer_secret_censored = self.consumer_secret[:4] + "*" * 10 + self.consumer_secret[-4:]
        access_token_censored = self.access_token[:4] + "*" * 10 + self.access_token[-4:]
        access_token_secret_censored = self.access_token_secret[:4] + "*" * 10 + self.access_token_secret[-4:]
        st.title("Crawling Data Twitter")
        st.subheader("Token Information")
        st.write("Consumer Key:", consumer_key_censored)
        st.write("Consumer Secret:", consumer_secret_censored)
        st.write("Access Token:", access_token_censored)
        st.write("Access Token Secret:", access_token_secret_censored)
        st.write('#### **Masukan Keyword**')
        keyword = st.text_input("Masukkan Keyword Yang Akan Dicari:")
        st.write('#### **Masukan Jumlah Data Yang Ingin Dicari**')
        length = st.number_input("Masukkan Jumlah Tweet Yang Akan Dicari:", min_value=1, step=1)
        if st.button("Run Crawl:mag:"):
            self.search_tweet(keyword, length)
            
# import library unutk stopword
from nltk.corpus import stopwords
# import library untuk stemming
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# import library untuk regex
import re
# import library untuk string
import string
# import library untuk tweet tokenizer
from nltk.tokenize import TweetTokenizer

@st.cache_resource()
class Preprocessing():
    def __init__(self, data):
        self.stop_words = stopwords.words('indonesian')
        self.factory = StemmerFactory()
        self.stemmer = self.factory.create_stemmer()
        self.dataframe = data
        self.slang_datasets = [
            'https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/master/new_kamusalay.csv',
            'https://raw.githubusercontent.com/panggi/pujangga/master/resource/formalization/formalizationDict.txt',
            'https://raw.githubusercontent.com/agusmakmun/SentiStrengthID/master/id_dict/slangword.txt',
            'https://raw.githubusercontent.com/ramaprakoso/analisis-sentimen/master/kamus/kbba.txt'
        ]
        more_slang = pd.DataFrame({
            'slang': ['sblm', 'sdh', 'laah', 'jlas', 'mnjlaskn', 'bhw', 'mmc', 'islamkaffah', 'serbaserbi', 'amp', 'baikm', 'berdalihminta',
                      'ngak', 'smbil', 'bnyak', 'knpa', 'pnmbakan', 'trjadi', 'klapangan', 'memnyemangati', 'gkda', 'bnyak', 'dngan', 'pndkung',
                      'krusuhan', 'bamnyaknya', 'mnyemangati', 'dkira', 'krnya', 'tmbakn', 'suporter'
                      ], 
            'normal': ['sebelum', 'sudah', 'lah', 'jelas', 'menjelaskan', 'bahwa', '', '', '', '', 'baik', 'berdalih minta','tidak', 
                       'sambil', 'banyak', 'kenapa', 'penembakan', 'terjadi', 'lapangan', 'menyemangati', 'tidak ada', 'banyak', 'dengan', 'pendukung',
                       'kerusuhan', 'banyaknya', 'menyemangati', 'dikira', 'karena', 'tembakan', 'supporter'
                       ]
        })
        self.slang_words = pd.concat([pd.read_csv(slang, sep='\t|:', header=None, names=['slang', 'normal'], engine='python', encoding='unicode_escape') for slang in self.slang_datasets], ignore_index=True)
        self.slang_words = pd.concat([self.slang_words, more_slang], ignore_index=True)
        self.slang_dict = dict(self.slang_words.drop_duplicates(subset='slang', keep='first').values)
        self.date = dt.datetime.now().strftime("%d-%m-%Y_%H-%M")
    
    def cleansing(self):
        # Membuang kolom tanggal
        if 'user_name' in self.dataframe.columns:
            self.dataframe.drop(['user_name'], axis=1, inplace=True)
        if 'Unnamed: 0' in self.dataframe.columns:
            self.dataframe.drop(['Unnamed: 0'], axis=1, inplace=True)
        self.dataframe.drop_duplicates(subset ="teks", keep = 'first', inplace = True)
        self.dataframe.dropna(inplace=True)
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
            #remove characters between words
            teks = re.sub(r'(?<=[a-zA-Z])\W+(?=[a-zA-Z])', ' ', teks)
            # remove punctuation marks and symbols
            teks = re.sub(r'[^\w\s]', '', teks)
            # remove underscore
            teks = re.sub(r'_', '', teks)
            # remove extra whitespace
            teks = re.sub(r'\s+', ' ', teks)
            teks = teks.strip()
            return teks
        self.dataframe['tweet_regex'] = self.dataframe['teks'].apply(text_prep_norm)
        return self.dataframe
    
    def normalization(self):
        def normalize_text(teks):
            # Replace slang words
            teks = ' '.join([self.slang_dict.get(word, word) for word in teks.split()])
            
            # Remove all non-alphanumeric characters and extra spaces
            teks = re.sub(r'[^\w\s]+', '', teks).strip().lower()  
            return teks
        self.dataframe['tweet_normal'] = self.dataframe['tweet_regex'].apply(normalize_text)
        return self.dataframe
    
    def stemming_stopword(self):
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
        def text_clean(teks):
            tokens = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True).tokenize(teks)
            clean_tokens = [t for t in tokens if t not in self.stop_words and t not in emoticons and t not in string.punctuation and len(t) > 1]
            return clean_tokens
        def text_stem(teks):
            stems = [self.stemmer.stem(t) for t in teks]
            return stems
        self.dataframe['tweet_clean'] = self.dataframe['tweet_normal'].apply(text_clean).apply(text_stem)
        return self.dataframe
    
    def join_words(self):
        def remove_punct(text):
            text  = " ".join([char for char in text if char not in string.punctuation])
            return text
        self.dataframe['tweet'] = self.dataframe['tweet_clean'].apply(lambda x: remove_punct(x))
        return self.dataframe
    
    def save(self):
        filename = f"Preprocessed_Dataset_{self.date}.csv"
        save_csv = self.dataframe.drop(['teks', 'tweet_regex', 'tweet_normal'], axis=1)
        csv = convert_df(save_csv)
        st.download_button(
            label="Download Dataset CSV:arrow_down:",
            data=csv,
            file_name=filename,
            mime='text/csv',
        )
        
#Import Visualization
import matplotlib.pyplot as plt
from wordcloud import WordCloud

@st.cache_resource()
class SentimentAnalyzer:
    def __init__(self, dataSet):
        self.dataSet = dataSet
        self.neg_words = None
        self.pos_words = None
        self.date = dt.datetime.now().strftime("%d-%m-%Y_%H-%M")
    
    def hit_len_word(self):
        self.dataSet['len_kalimat'] = self.dataSet['tweet'].astype(str).apply(lambda x: len(x.split()))
        return self.dataSet

    def read_lex(self):
        file_neg = 'https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv'
        file_pos = 'https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv'
        self.neg_words = pd.read_csv(file_neg, sep='\t')
        self.pos_words = pd.read_csv(file_pos, sep='\t')
        return self.neg_words, self.pos_words

    def get_sen_score_weight(self):
        self.neg_words, self.pos_words = self.read_lex()
        def sent_score(kalimat, lexicon):
            score = 0
            words = kalimat.split()
            for word in words:
                if word in lexicon['word'].values:
                    x = lexicon.loc[lexicon['word'] == word, 'weight']
                    score += int(x.values[0])
            return score
        num_pos = [sent_score(i, self.pos_words) for i in self.dataSet['tweet'].astype(str)]
        self.dataSet['pos_count'] = num_pos
        num_neg = [sent_score(i, self.neg_words) for i in self.dataSet['tweet'].astype(str)]
        self.dataSet['neg_count'] = num_neg
        self.dataSet['sentiment'] = round((self.dataSet['pos_count'] - ((self.dataSet['neg_count'])*-1)) / self.dataSet['len_kalimat'], 4)
        self.dataSet['label'] = self.dataSet['sentiment'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
        self.dataSet = self.dataSet.dropna()
        return self.dataSet
    
    def get_sen_score_sum(self):
        self.neg_words, self.pos_words = self.read_lex()
        def sent_score(kalimat, lex_words):
            score = 0
            words = kalimat.split()
            for i in words:
                if i in lex_words:
                    score += 1
            return score
        
        pos_words_set = set(self.pos_words['word'].values)
        neg_words_set = set(self.neg_words['word'].values)

        pos_counts = [sent_score(kalimat, pos_words_set) for kalimat in self.dataSet['tweet'].astype(str)]
        neg_counts = [sent_score(kalimat, neg_words_set) for kalimat in self.dataSet['tweet'].astype(str)]
        self.dataSet['pos_count'] = pos_counts
        self.dataSet['neg_count'] = neg_counts

        self.dataSet['sentiment'] = round((self.dataSet['pos_count'] - self.dataSet['neg_count']) / self.dataSet['len_kalimat'], 4)
        self.dataSet['label'] = self.dataSet['sentiment'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
        self.dataSet = self.dataSet.dropna()    
        return self.dataSet
    
    def show_label(self):
        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct*total/100.0))
                return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
            return my_autopct

        fig, ax = plt.subplots(figsize=(5,5))
        self.dataSet['label'].value_counts().plot(kind='pie', title='Jumlah Label', autopct=make_autopct(self.dataSet['label'].value_counts()), ax=ax)
        st.pyplot(fig, use_column_width=False)

    def save_label(self,method):
        filename = f"Labelled_{method}_Dataset_{self.date}.csv"
        save_csv = self.dataSet.drop(['len_kalimat','pos_count','neg_count','sentiment'], axis=1)
        csv = convert_df(save_csv)
        st.download_button(
            label="Download Dataset CSV:arrow_down:",
            data=csv,
            file_name=filename,
            mime='text/csv',
        )
        
    def display_wordcloud(self, label):
        tweet_subset = self.dataSet[self.dataSet['label'] == label]
        tweet_subset['tweet'] = tweet_subset['tweet'].astype(str)
        if tweet_subset.empty:
            st.warning("No tweets available for the selected sentiment.")
            return
        else:
            color = 'Greens' if label == 'positive' else ('Reds' if label == 'negative' else 'Blues')
            wordcloud = WordCloud(width=800, height=800, background_color='black', min_font_size=10, colormap=color).generate(' '.join(tweet_subset['tweet']))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout(pad=0)
            st.pyplot(use_column_width=True)



import numpy as np
from collections import Counter

class TFIDFVectorizer:
    def __init__(self):
        self.vocab = {}
        self.num_docs = 0
        self.idf = {}
    
    def fit_transform(self, corpus):
        # Compute the term frequency (TF) for each word in each document
        tf = []
        for doc in corpus:
            word_counts = Counter(doc.split())
            max_count = max(word_counts.values())
            tf.append({word: count / max_count for word, count in word_counts.items()})
        
        # Compute the inverse document frequency (IDF) for each word in the corpus
        self.num_docs = len(corpus)
        for doc in corpus:
            for word in set(doc.split()):
                if word in self.idf:
                    self.idf[word] += 1
                else:
                    self.idf[word] = 1
        
        for word in self.idf:
            self.idf[word] = np.log(self.num_docs / self.idf[word])
        
        # Compute the TF-IDF score for each word in each document
        tf_idf = []
        for i, doc in enumerate(corpus):
            tf_idf.append({})
            for word in set(doc.split()):
                tf_idf[i][word] = tf[i][word] * self.idf[word]
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
        
        # Convert the TF-IDF scores to a Pandas DataFrame
        data = []
        for i, doc in enumerate(corpus):
            row = [0] * len(self.vocab)
            for word in set(doc.split()):
                j = self.vocab[word]
                row[j] = tf_idf[i][word]
            data.append(row)
        df = pd.DataFrame(data, columns=self.vocab.keys())
        
        # Compute the TF, IDF, and TF-IDF scores for each term in each document
        term_data = []
        for i, doc in enumerate(corpus):
            term_counts = Counter(doc.split())
            max_count = max(term_counts.values())
            for term, count in term_counts.items():
                tf = count / max_count
                idf = self.idf.get(term, 0)
                tf_idf = tf * idf
                term_data.append((i, term, tf, idf, tf_idf))
        term_df = pd.DataFrame(term_data, columns=["doc_id", "term", "tf", "idf", "tf_idf"])
        
        # Return both the DataFrames and a NumPy array
        return df, term_df, df.values

    def transform(self, documents):
        # Convert each document into a TF-IDF vector
        num_documents = len(documents)
        num_features = len(self.vocab)
        vectors = np.zeros((num_documents, num_features))
        
        for i, document in enumerate(documents):
            words = str(document).split()
            word_counts = Counter(words)
            max_freq = max(word_counts.values())
            for word, count in word_counts.items():
                if word in self.vocab:
                    j = self.vocab[word]
                    tf = 0.5 + 0.5 * count / max_freq  # Use smoothed term frequency
                    tf_idf = tf * self.idf[word]
                    vectors[i, j] = tf_idf
        
        # Convert the TF-IDF vectors to a Pandas DataFrame
        df = pd.DataFrame(vectors, columns=self.vocab.keys())
        return df, vectors

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
import pickle
import io

class DataMiningSVM:
    def __init__(self, kernel='linear', C=1.0, gamma='auto', probability=True, random_state=0, verbose=False, 
                 max_iter=-1, shrinking=True, decision_function_shape='ovr', break_ties=False):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.probability = probability
        self.random_state = random_state
        self.verbose = verbose
        self.max_iter = max_iter
        self.shrinking = shrinking
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.target_encoder = LabelEncoder()
        self.vectorizer = TFIDFVectorizer()
        self.counter = Counter()
        self.smt = SMOTE()
        self.clf = svm.SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, probability=self.probability,
                            random_state=self.random_state, verbose=self.verbose, max_iter=self.max_iter,
                            shrinking=self.shrinking, decision_function_shape=self.decision_function_shape,
                            break_ties=self.break_ties,class_weight='balanced')
        self.date = dt.datetime.now().strftime("%d-%m-%Y_%H-%M")
    
    def train_test_split(self, df, test_size=0.25, random_state=42):
        df["target"] = self.target_encoder.fit_transform(df["label"])
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        return train_data, test_data
    
    def vectorize(self, train_data):
        train_df, term_df, X_train = self.vectorizer.fit_transform(train_data["tweet"])
        return train_df, term_df,X_train
    
    def oversample(self, X_train, y_train):
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                self.counter = Counter(y_train)
                st.write('Before oversampling: \n\n'
                        'Positive: {}, Negative: {}, Neutral: {}'.format(self.counter[1], self.counter[0], self.counter[2]))
                X_train_sm, y_train_sm = self.smt.fit_resample(X_train, y_train)
            with col2:
                self.counter = Counter(y_train_sm)
                st.write('\nAfter oversampling: \n\n'
                        'Positive: {} \n, Negative: {}, Neutral: {}'.format(self.counter[1], self.counter[0], self.counter[2]))
        return X_train_sm, y_train_sm

    def fit(self, X_train_sm, y_train_sm):
        self.clf.fit(X_train_sm, y_train_sm)
        st.write("Model trained successfully!")
            
    def save_model(self):
        model_filename = "SVMmodel_{}.pkl".format(self.date)
        vectorizer_filename = "TF_IDFvectorizer_{}.pkl".format(self.date)
        # Read the model and vectorizer objects into memory
        model_bytes = pickle.dumps(self.clf)
        vectorizer_bytes = pickle.dumps(self.vectorizer)
        st.write("Model saved as \n{} \n\n Vectorizer saved as \n{}".format(model_filename, vectorizer_filename))
        # Download the files using st.download
        model_buf = io.BytesIO(model_bytes)
        vectorizer_buf = io.BytesIO(vectorizer_bytes)
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(label="Download Model:arrow_down:", data=model_buf, file_name=model_filename, mime='application/octet-stream')
            with col2:    
                st.download_button(label="Download Vectorizer:arrow_down:", data=vectorizer_buf, file_name=vectorizer_filename, mime='application/octet-stream')

    def evaluation(self, test_data):
        test_df, X_test = self.vectorizer.transform(test_data["tweet"])
        y_test = test_data["target"]
        y_pred = self.clf.predict(X_test)
        accuracy = self.clf.score(X_test, y_test)
        precision = metrics.precision_score(y_test, y_pred, average='weighted')
        recall = metrics.recall_score(y_test, y_pred, average='weighted')
        f1_score = metrics.f1_score(y_test, y_pred, average='weighted')
        st.write("## Accuracy:", accuracy)
        st.write("## Precision:", precision)
        st.write("## Recall:", recall)
        st.write("## F1 Score:", f1_score)
        return y_test, y_pred
    
    def display(self, y_test, y_pred):
        cm = metrics.confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=["Negative", "Positive", "Neutral"], 
                    yticklabels=["Negative", "Positive", "Neutral"])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel("Predicted label", fontsize=14)
        plt.ylabel("True label", fontsize=14)
        plt.title("Confusion Matrix", fontsize=16)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        
    def display_report(self, y_test, y_pred):
        # Display classification report as table
        report = metrics.classification_report(y_test, y_pred, zero_division=1, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.table(report_df)

    
    def predict_all(self, df):
        X_tfidf, _ = self.vectorizer.transform(df["tweet"])
        y_pred = self.clf.predict(X_tfidf)
        df["predicted_label"] = self.target_encoder.inverse_transform(y_pred)
        
        # Show the DataFrame
        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct*total/100.0))
                return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
            return my_autopct
        fig, ax = plt.subplots(figsize=(5,5))
        df['predicted_label'].value_counts().plot(kind='pie', title='Jumlah Label', autopct=make_autopct(df['predicted_label'].value_counts()), ax=ax)
        st.pyplot(fig, use_column_width=False)
        st.dataframe(df)
    
import pickle
from io import BytesIO

class SentimentPredictor:
    def __init__(self, model_file, vectorizer_file):
        # Read the model and vectorizer from the uploaded file
        model_buf = BytesIO(model_file.read())
        vectorizer_buf = BytesIO(vectorizer_file.read())
        self.clf = pickle.load(model_buf)
        self.vectorizer = pickle.load(vectorizer_buf)

    def predict(self, data):
        data.astype(str)
        X_tfidf, _ = self.vectorizer.transform(data["tweet"])
        y_pred = self.clf.predict(X_tfidf)

        # Convert the predicted labels to human-readable sentiment values
        sentiment_labels = {0: 'Negative', 2: 'Neutral', 1: 'Positive'}
        data["label"] = [sentiment_labels[label] for label in y_pred]
        return data
    def pie_chart(self, data):
        # Show the DataFrame
        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct*total/100.0))
                return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
            return my_autopct
        fig, ax = plt.subplots(figsize=(5,5))
        data['label'].value_counts().plot(kind='pie', title='Jumlah Label', autopct=make_autopct(data['label'].value_counts()), ax=ax)
        st.pyplot(fig, use_column_width=False)


