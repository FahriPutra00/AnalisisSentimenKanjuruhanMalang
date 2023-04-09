import streamlit as st
from streamlit_option_menu import option_menu
from Class_Program import *
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Analisis Sentimen",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "Analisis Sentimen Kanjuruhan Malang"
    }
)

with st.sidebar:
    selected = option_menu("Menu",["Crawling Data","Preprocessing","Labelling","Analisis Sentimen", "Prediksi Sentimen"],
                           icons=['search','blockquote-left', 'tags', 'gear', 'file-earmark-arrow-up'], menu_icon="cast",
                           default_index=0, styles={
        "container": {"padding": "5!important", "padding-top":"0px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"5px"},
    })



if selected =="Crawling Data":
    crawler = TwitterCrawler()
    crawler.run()

if selected =='Preprocessing':   
    st.title("Preprocessing Dataset")
    # Upload multiple CSV files
    uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)
    # Merge all CSV files into a single dataframe
    if uploaded_files:
        try:
            dataframes = [pd.read_csv(file) for file in uploaded_files]
            merged_df = pd.concat(dataframes, ignore_index=True)
            # Display merged dataframe
            st.subheader("Merged CSV files")
            st.write(f"Jumlah kolom data: {merged_df.shape[0]}")
            st.dataframe(merged_df, width=None, height=None, use_container_width=True)
        except pd.errors.EmptyDataError:
            st.error("Error: Empty CSV file(s) uploaded.")
        except Exception as e:
            st.error(f"Error: {e}")
        # Initialize Preprocessing object
        preprocessor = Preprocessing(merged_df)
        # Cleansing dataframe
        with st.spinner('Running Data Cleansing...'):
            data_clean = preprocessor.cleansing()
        st.subheader("Data Cleansing")
        st.dataframe(data_clean, width=None, height=None, use_container_width=True)
        with st.spinner('Running Data Normalization...'):
            data_normal = preprocessor.normalization()
        st.subheader("Data Normalization")
        data_normal_vis = data_normal.copy()
        data_normal_vis.drop(['teks'], axis=1, inplace=True)
        st.dataframe(data_normal_vis, width=None, height=None, use_container_width=True)
        with st.spinner('Running Data Stemming...'):
            data_token = preprocessor.stemming_stopword()
        st.subheader("Data Stemming, Stopword, dan Tokenize")
        data_token_vis = data_token.copy()
        data_token_vis.drop(['teks','tweet_regex'], axis=1, inplace=True)
        st.dataframe(data_token_vis, width=None, height=None, use_container_width=True)
        with st.spinner('Running Data Joining...'):
            data_cleansing = preprocessor.join_words()
        st.subheader("Remove Punctuation & Join Words")
        data_cleansing_vis = data_cleansing.copy()
        data_cleansing_vis.drop(['teks','tweet_regex','tweet_normal'], axis=1, inplace=True)
        st.dataframe(data_cleansing_vis, width=None, height=None, use_container_width=True)
        st.dataframe(data_cleansing, width=None, height=None, use_container_width=True)
        st.write(f"Jumlah kolom data: {data_cleansing.shape[0]}")
        preprocessor.save()
        
if selected == 'Labelling':
    st.title("Labelling Dataset")
    st.write("#### Upload Data Preprocessed")
    uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=False)
    if uploaded_files is not None:
        st.write("#### Pilih Metode Labelling Dataset")
        labelling_method = st.radio("Select labelling method:", ["Sum Labelling","Weight Labelling"], index=0,horizontal=True, key='labelling_method')
        data_clean = pd.read_csv(uploaded_files)
        labelling = SentimentAnalyzer(data_clean)
        if labelling_method == "Sum Labelling":
            # perform sum labelling
            st.dataframe(data_clean, width=None, height=None, use_container_width=True)
            with st.spinner('Running Hitung Panjang Kalimat...'):
                data_hit = labelling.hit_len_word()
            st.subheader("Hitung Panjang Kalimat")
            st.dataframe(data_hit, width=None, height=None, use_container_width=True)
            with st.spinner('Running Hitung Sentiment Score...'):
                data_score = labelling.get_sen_score_sum()
            st.subheader("Hitung Sentimen Score Weight")
            st.dataframe(data_score, width=None, height=None, use_container_width=True)
            # set the width of the container
            with st.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    with st.spinner('Running Hitung WordCloud...'):
                        st.write("# Positive Wordcloud")
                        labelling.display_wordcloud("positive")
                with col2:
                    with st.spinner('Running Hitung WordCloud...'):
                        st.write("# Negative Wordcloud")
                        labelling.display_wordcloud("negative")
                with col3:
                    with st.spinner('Running Hitung WordCloud...'):
                        st.write("# Neutral Wordcloud")
                        labelling.display_wordcloud("neutral")
            with st.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.empty()
                with col2:
                    with st.spinner('Running Hitung Sentiment Score...'):
                        st.write("#### Hasil Labelling Dataset")
                        labelling.show_label()
                        labelling.save_label("Sum")
                with col3:
                    st.empty()
        elif labelling_method == "Weight Labelling":
            # perform weight labelling
            st.dataframe(data_clean, width=None, height=None, use_container_width=True)
            with st.spinner('Running Hitung Panjang Kalimat...'):
                data_hit = labelling.hit_len_word()
            st.subheader("Hitung Panjang Kalimat")
            st.dataframe(data_hit, width=None, height=None, use_container_width=True)
            with st.spinner('Running Hitung Sentiment Score...'):
                data_score = labelling.get_sen_score_weight()
            st.subheader("Hitung Sentimen Score Weight")
            st.dataframe(data_score, width=None, height=None, use_container_width=True)
            # set the width of the container
            with st.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    with st.spinner('Running Hitung WordCloud...'):
                        st.write("# Positive Wordcloud")
                        labelling.display_wordcloud("positive")
                with col2:
                    with st.spinner('Running Hitung WordCloud...'):
                        st.write("# Negative Wordcloud")
                        labelling.display_wordcloud("negative")
                with col3:
                    with st.spinner('Running Hitung WordCloud...'):
                        st.write("# Neutral Wordcloud")
                        labelling.display_wordcloud("neutral")
            with st.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.empty()
                with col2:
                    with st.spinner('Running Hitung Sentiment Score...'):
                        st.write("# Hasil Labelling Dataset")
                        labelling.show_label()
                        labelling.save_label("Weight")
                with col3:
                    st.empty()


if selected =="Analisis Sentimen":   
    st.title("Analisis Sentimen")
    st.write("#### Upload Data Labelled")
    uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=False)
    if uploaded_files is not None:
        st.header("SVM Parameters")
        with st.container():
            col1, col2, = st.columns(2)
            with col1:
                kernel = st.selectbox("Kernel", ['linear', 'poly', 'rbf', 'sigmoid'])
                decision_function_shape = st.selectbox("Decision function shape", ['ovr', 'ovo'])
                max_iter = st.slider("Max iterations", -1, 10000, value=-1)
                C = st.slider("C", 0.1, 10.0, step=0.1)
            with col2:
                gamma = st.selectbox("Gamma", ['scale', 'auto'] + [i/10 for i in range(1, 11)])
                random_state = st.number_input("Random state", value=42)
                probability = st.checkbox("Probability", value=True)
                shrinking = st.checkbox("Shrinking", value=True)
                verbose = st.checkbox("Verbose", value=False)
                break_ties = st.checkbox("Break ties", value=False)
        st.write("#### Data Labelled")
        data_label = pd.read_csv(uploaded_files)
        st.dataframe(data_label, width=None, height=None, use_container_width=True)
        show_word = SentimentAnalyzer(data_label)
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                with st.spinner('Running Hitung WordCloud...'):
                    st.write("# Positive Wordcloud")
                    show_word.display_wordcloud("positive")
            with col2:
                with st.spinner('Running Hitung WordCloud...'):
                    st.write("# Negative Wordcloud")
                    show_word.display_wordcloud("negative")
            with col3:
                with st.spinner('Running Hitung WordCloud...'):
                    st.write("# Neutral Wordcloud")
                    show_word.display_wordcloud("neutral")
        dmsvm = DataMiningSVM(kernel=kernel, C=C, gamma=gamma, probability=probability, random_state=random_state,
                      verbose=verbose, max_iter=max_iter, shrinking=shrinking, 
                      decision_function_shape=decision_function_shape, break_ties=break_ties)
        # Train the SVM model
        st.header("Train Test Split Parameters")
        with st.container():
            col1, col2, = st.columns(2)
            with col1:
                test_size = st.slider("Test Size", 0.1, 0.5, 0.25, 0.05)
            with col2:
                random_state = st.number_input("Random State", 0, 100, 42, 1)
        st.subheader("Split Data")
        train_data, test_data = dmsvm.train_test_split(data_label, test_size=test_size, random_state=random_state)
        st.write("#### Data Train")
        st.dataframe(train_data, width=None, height=None, use_container_width=True)
        st.write("#### Data Test")
        st.dataframe(test_data, width=None, height=None, use_container_width=True)
        st.subheader("Pembobotan TF-IDF")
        train_df,term_df, X_train = dmsvm.vectorize(train_data)
        st.dataframe(train_df, width=None, height=None, use_container_width=True)
        st.write("#### Nilai TF-IDF")
        st.dataframe(term_df, width=None, height=None, use_container_width=True)
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.write("# Fitting Model")
                st.write("Ceklis untuk melakukan oversampling")
                # create checkbox to toggle oversampling
                oversample = st.checkbox('Oversample training data', value=True)
                # apply oversampling if checkbox is checked
                if oversample:
                    X_train_sm, y_train_sm = dmsvm.oversample(X_train, train_data["target"])
                    dmsvm.fit(X_train_sm, y_train_sm)
                else:
                    dmsvm.fit(X_train, train_data["target"])
            with col2:
                st.write("# Hasil Evaluasi Model")
                y_test, y_pred= dmsvm.evaluation(test_data)
        with st.container():
            col1, col2, = st.columns(2)
            with col1:
                dmsvm.display(y_test, y_pred)
            with col2:
                df = dmsvm.predict_all(test_data)
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.empty()
            with col2:
                st.write("## Save Model SVM")
                dmsvm.save_model()
            with col3:
                st.empty()
    

if selected =="Prediksi Sentimen":
    st.title("Prediksi Sentimen")   
    st.write("## Upload Model SVM")
    # Upload pickle file
    uploaded_model = st.file_uploader("Upload Model SVM (.pkl)", type=["pkl"])
    uploaded_vectorizer = st.file_uploader("Upload Vectorizer TF-IDF (.pkl)", type=["pkl"])
    if uploaded_model and uploaded_vectorizer is not None:
        pred = SentimentPredictor(uploaded_model, uploaded_vectorizer)
        uploaded_file = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=False)
        if uploaded_file is not None:
            with st.spinner('Running Predict New Data...'):
                data_predict = pd.read_csv(uploaded_file)
                st.dataframe(data_predict, width=None, height=None, use_container_width=True)
                df_pred = pred.predict(data_predict)
                st.dataframe(df_pred, width=None, height=None, use_container_width=True)
                # show_wordc = SentimentAnalyzer(df_pred)
                # with st.container():
                #     col1, col2, col3 = st.columns(3)
                #     with col1:
                #         with st.spinner('Running Hitung WordCloud...'):
                #             st.write("# Positive Wordcloud")
                #             show_wordc.display_wordcloud("positive")
                #     with col2:
                #         with st.spinner('Running Hitung WordCloud...'):
                #             st.write("# Negative Wordcloud")
                #             show_wordc.display_wordcloud("negative")
                #     with col3:
                #         with st.spinner('Running Hitung WordCloud...'):
                #             st.write("# Neutral Wordcloud")
                #             show_wordc.display_wordcloud("neutral")
                with st.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.empty()
                    with col2:
                        pred.pie_chart(df_pred)
                    with col3:
                        st.empty()
    elif uploaded_model is not None and uploaded_vectorizer is None:
        st.write("## Upload Model SVM")
    elif uploaded_vectorizer is not None and uploaded_model is None:
        st.write("## Upload Vectorizer TF-IDF")
    else:
        st.write("## Upload Model SVM dan Vectorizer TF-IDF")


hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """

st.markdown(hide_st_style, unsafe_allow_html=True)