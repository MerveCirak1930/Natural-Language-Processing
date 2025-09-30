import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


################## Veri Seti Oluşturma ####################

# Veri seti, https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset ' den alınmıştır.

# veri seti yukle
df = pd.read_csv("4-Frequency-Based Methods/TF_IDF/sms_spam.csv")


#################### Veri Temizleme ve Tokenleştirme ####################
import re

# Veri temizleme fonksiyonu
def clean_text(text):

    # Küçük harfe dönüştürme
    text = text.lower()

    # Sayıları kaldırma
    text = re.sub(r'\d+', '', text)

    # Noktalama işaretlerini kaldırma
    text = re.sub(r'[^\w\s]', '', text)

    # Kısa kelimeleri (2 karakterden az) kaldırma
    text = ' '.join([word for word in text.split() if len(word) > 2])

    # Stop-word'leri temizle
    import nltk
    from nltk.corpus import stopwords

    # Eğer stopwords veri seti eksikse indir
    #nltk.download("stopwords")

    # Türkçe stop-words listesini yükle
    stop_word_tr = set(stopwords.words("turkish"))
    #print(stop_word_tr)
    text = [kelime for kelime in text.split() if kelime not in stop_word_tr]

    return ' '.join(text)

# Metinleri temizle
df['cleaned_text'] = df['text'].apply(clean_text)

# Temizlenmiş metnin ilk birkaç satırına bakalım
#print(df['cleaned_text'].head())


#################### TF -IDF ile Özellik Çıkartımı  ####################

# TF-IDF vektörleştirme (L2 normalizasyonu ile)
vectorizer = TfidfVectorizer(norm='l2')

X = vectorizer.fit_transform(df["cleaned_text"])

# kelime kumesi
feature_names = vectorizer.get_feature_names_out()

# Feature ların ortalama TF-IDF skorlarının hesaplanması
tfidf_score_ort = X.mean(axis=0).A1 # ortalama tf-idf degerleri

df_tfidf = pd.DataFrame({"word":feature_names, "tfidf_score_ort":tfidf_score_ort})
#print(df_tfidf)

df_tfidf_sorted = df_tfidf.sort_values(by = "tfidf_score_ort", ascending=False)
print(df_tfidf_sorted)

