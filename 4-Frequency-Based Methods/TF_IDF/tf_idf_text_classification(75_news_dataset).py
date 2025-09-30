import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


################## Veri Seti Oluşturma ####################

# Kaynak: http://www.kemik.yildiz.edu.tr/veri_kumelerimiz.html
# 	5 sinifa ait 15'er adet haber metni (ham ve arff formatinda)	75haber.zip klasörü indirilerek 
#   text ve label lar dataframe formatında düzenlenmiştir

folder_path = "4-Frequency-Based Methods/TF_IDF/raw_texts"

class_names = ['ekonomi', 'magazin', 'saglik','siyasi','spor']
data = []

for class_name in class_names:
    class_folder = os.path.join(folder_path, class_name)
    for file_name in os.listdir(class_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(class_folder, file_name)
            try:
                with open(file_path, 'r', encoding='ISO-8859-9') as file:
                    text = file.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin1') as file:
                    text = file.read()
            data.append([text, class_name])

df = pd.DataFrame(data, columns=['text', 'label'])

#################### Veri Temizleme ####################
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
print('temizlenmiş metin:\n',df['cleaned_text'].head())


#################### TF-IDF Özellik Çıkartımı ####################

# TF-IDF vektörleştirme (L2 normalizasyonu ile)
#vectorizer = TfidfVectorizer(norm='l2')

# TF-IDF Parametrelerini Optimize Etme
vectorizer = TfidfVectorizer(norm='l2', ngram_range=(1,2), min_df=2)

X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

#################### Model Eğitimi ####################
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#model = MultinomialNB()
#model = LogisticRegression(max_iter=1000)
#model = RandomForestClassifier(n_estimators=100, random_state=42)
model = SVC(kernel='linear', C=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#  SVM Modelini İyileştir -> Regularization (C) Değerini İncele
from sklearn.model_selection import GridSearchCV
# param_grid = {'C': [0.1, 1, 10, 100]}
# grid = GridSearchCV(SVC(kernel='linear'), param_grid, cv=5)
# grid.fit(X_train, y_train)
# print('best',grid.best_params_)


#print("Doğruluk: ", accuracy_score(y_test, y_pred))
print("\nSVM Sınıflandırma Raporu: \n", classification_report(y_test, y_pred))

#################### Model Test Etme ####################
user_input = input("Lütfen bir metin girin: ")
cleaned_input = clean_text(user_input)
vectorized_input = vectorizer.transform([cleaned_input])
predicted_label = model.predict(vectorized_input)[0]
print(f"Modelin tahmini: {predicted_label}")


# "Maç kötü bitti"	spor
#  " enflasyon yükseliyor "	siyasi
# " diyabet riski artıyor "	sağlık
