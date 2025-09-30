

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter


################## Veri Seti Oluşturma ####################

# IMDB Dataset.csv veri seti şu kaynaktan alınmıştır: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

df = pd.read_csv("4-Frequency-Based Methods/BoW/IMDB Dataset.csv")

df2 = df.head(100)

# Metin verileri alalım
documents = df["review"]
labels = df["sentiment"]  # positive /negative

#################### Veri Temizleme ve Tokenleştirme ####################

# Metinleri temizleme fonk.
def clean_text(text):

    # kucuk harf donusumu
    text = text.lower()

    # rakamları temizleme
    text = re.sub(r"\d+", "", text)

    # ozel karakterleri temizleme
    text = re.sub(r"[^\w\s]", "", text)

    # kisa kelimeleri temizleme
    text = " ".join([word for word in text.split() if len(word) > 2])

    # stop words leri temizleme
    import nltk
    from nltk.corpus import stopwords

    # Eğer stopwords veri seti eksikse indir
    #nltk.download("stopwords")

    # İngilizce stop-words listesini yükle
    stop_word_tr = set(stopwords.words("english"))

    # Stop-word'leri temizle
    text = [kelime for kelime in text.split() if kelime not in stop_word_tr]

    return ' '.join(text) 

# Metinleri temizleme 
cleaned_documents = [clean_text(doc) for doc in documents]


#################### BoW (Bag of Words) Özellik Çıkartımı ####################

# bow
vectorizer  = CountVectorizer()

# metni sayısal vektöre dönüstürme
X = vectorizer.fit_transform(cleaned_documents[:100])

# Kelime kümesi 
feature_names = vectorizer.get_feature_names_out()
#print("Kelime kümesi:",feature_names) #

# Vektör temsili
print("Vektör temsili:",X.toarray())

vektor_temsili2 = X.toarray()[:2]
#print("daraltılmıs vektor temsili",vektor_temsili2)


# Vektor temsili dataframe
df_bow = pd.DataFrame(X.toarray(), columns = feature_names)
print('\n\n\nVektor temsilinin dataframe olarak gösterilmesi:')
print(df_bow[:10])

# Kelime frekansı
word_counts = X.sum(axis = 0).A1
word_freq = dict(zip(feature_names,word_counts))
print('\n\n\nKelimeler ve frekansları:')
print(word_freq)
 
# En yaygın kullanılan ilk 5 kelime
most_common_5_words = Counter(word_freq).most_common(5)
print('\n\n\n En sık kullanılan 5 kelime',most_common_5_words)  # stop words leri kaldırmadan önce: [('the', 1378), ('and', 645), ('this', 258), ('that', 236), ('was', 181)]
                            # stop words leri kaldırdıktan sonra:[('movie', 169), ('film', 127), ('one', 100), ('like', 80), ('even', 58)]