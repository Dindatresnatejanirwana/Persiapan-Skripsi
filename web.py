from flask import Flask, render_template, request
import csv
import re
import string
import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

nltk.download('punkt')
nltk.download('stopwords')

dataset_path = 'static/data.csv'

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/form", methods=["GET", "POST"])
def form():
    if request.method == "POST":
        nama_mahasiswa = request.form.get("nama")
        program_studi = request.form.get("prodi")
        judul_penelitian = request.form.get("judul")
        
        if judul_penelitian:
            with open(dataset_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['none', judul_penelitian])
            
        df=pd.read_csv(dataset_path)
        df=df[~df['PEMBIMBING'].str.contains('none') | df.index.isin([df.index[-1]])]
        df= df[['PEMBIMBING', 'JUDUL TUGAS AKHIR/PENELITIAN']]
        df=df[['PEMBIMBING', 'JUDUL TUGAS AKHIR/PENELITIAN']].rename(columns={'PEMBIMBING': 'NAMA DOSEN', 'JUDUL TUGAS AKHIR/PENELITIAN': 'TEXT'})
        df['lowercase']=df['TEXT'].str.lower()
        df['number']=df['lowercase'].apply(lambda x: re.sub(r'\d+', '', x))
        df['punctuatio']=df['number'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        df['whitespace']=df['punctuatio'].str.strip()
        df['tokenize']=df['whitespace'].apply(lambda x: word_tokenize(x))
        replacement0 = {"Bas":" Basis","convolut":"Convolutional","websit":"website","juic":"juice","jeni":"jenis","metod":"metode","sdmstrategi":"sdm strategi",
                "bisni":"bisnis","ancang":"rancang","chang":"change","kelulus":"lulus","analysi":"analys","kualita":"kualitas","onlin":"online",
                "â€œXGRACIASâ€":"xgracias","gun":"guna","knowledg":"knowledge","foundat":"foundation","Customerâ€™s":"customer","readersâ€™":"reader",
                "System â€™":"system","OLEH â€“ OLEH":"oleh oleh","UTAUTâ€“ ":"utaut"}
        df['replacement'] = df['tokenize'].apply(lambda x: [replacement0.get(word, word) for word in x])
        factory = StopWordRemoverFactory()
        stopwords_in = factory.get_stop_words()
        additional_stopwords_in = ['xyz','guna']
        stopwords_in += additional_stopwords_in
        df['stopwords_in'] = df['replacement'].apply(lambda x: [word for word in x if word not in stopwords_in])
        stopwords_en = stopwords.words('english')
        additional_stopwords_en = ['iso']
        stopwords_en += additional_stopwords_en
        df['stopwords_en'] = df['stopwords_in'].apply(lambda x: [word for word in x if word not in stopwords_en])
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        df['stemming_id'] = df['stopwords_en'].apply(lambda x: [stemmer.stem(word) for word in x])
        def snowball_stemming(words):
            stemmer = SnowballStemmer("english")
            stemmed_words = [stemmer.stem(word) for word in words]
            return stemmed_words
        df['stemming_en'] = df['stemming_id'].apply(snowball_stemming)
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(df['stemming_en'].apply(' '.join))
        tfidf = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names_out())
        result = pd.concat([df['NAMA DOSEN'], tfidf], axis=1)
        training_data = np.nan_to_num(result.iloc[:-1, 1:].values, nan=0.0)
        training_labels = np.nan_to_num(result.iloc[:-1, 0].values, nan=0.0)
        testing_data = np.nan_to_num(result.iloc[-1, 1:].values.reshape(1, -1), nan=0.0)
        naive_bayes = MultinomialNB()
        naive_bayes.fit(training_data, training_labels)
        predicted_category = naive_bayes.predict(testing_data)
        posterior_probs = naive_bayes.predict_proba(testing_data)[0]
        sorted_probs = sorted(enumerate(posterior_probs), key=lambda x: x[1], reverse=True)
        X1 = naive_bayes.classes_[sorted_probs[0][0]]
        X2 = str(round(posterior_probs[sorted_probs[0][0]] * 100, 2)) + "%"
        Y1 = naive_bayes.classes_[sorted_probs[1][0]]
        Y2 = str(round(posterior_probs[sorted_probs[1][0]] * 100, 2)) + "%"
            
        return render_template("hasil.html", nama_mahasiswa=nama_mahasiswa, program_studi=program_studi, judul_penelitian=judul_penelitian, X1=X1, X2=X2, Y1=Y1, Y2=Y2)
    return render_template("form.html")

@app.route("/hasil")
def hasil():
    return render_template("hasil.html")

if __name__ == "__main__":
    app.run(debug=True)