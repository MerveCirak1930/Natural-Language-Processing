import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

################## Veri Seti Oluşturma ####################

# Veri seti, https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset ' den alınmıştır.

# veri seti yukle
df = pd.read_csv("4-Frequency-Based Methods/TF_IDF/sms_spam.csv")

#################### Veri Temizleme ####################
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
    stop_word_tr = set(stopwords.words("english"))
    #print(stop_word_tr)
    text = [kelime for kelime in text.split() if kelime not in stop_word_tr]

    return ' '.join(text)

# Metinleri temizle
df['cleaned_text'] = df['text'].apply(clean_text)
print('Temizlenmis df:\n',df['cleaned_text'].head())

#################### TF-IDF Özellik Çıkartımı ####################

# TF-IDF vektörleştirme (L2 normalizasyonu ile)
vectorizer = TfidfVectorizer(norm='l2')

X = vectorizer.fit_transform(df['cleaned_text'])
y = df['type']

#################### Model Eğitimi ####################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Doğruluk: ", accuracy_score(y_test, y_pred))
print("MNB Sınıflandırma Raporu: \n", classification_report(y_test, y_pred))

#################### Model Test Etme ####################
user_input = input("Lütfen bir film yorumu girin: ")
cleaned_input = clean_text(user_input)
vectorized_input = vectorizer.transform([cleaned_input])
predicted_label = model.predict(vectorized_input)[0]
print(f"Modelin tahmini: {predicted_label}")
