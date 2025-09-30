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

# Kaynak: http://www.kemik.yildiz.edu.tr/veri_kumelerimiz.html
# 	3 sinifa (pozitif, negatif, tarafsiz) ait 105 film kritigi (ham ve arff formatinda)	film_yorumlari.zip klasörü indirilerek 
#   text ve label lar dataframe formatında düzenlenmiştir

folder_path = "4-Frequency-Based Methods/TF_IDF/raw_texts"
class_names = ['pozitif', 'negatif', 'tarafsız']
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
nltk.download("stopwords")
stop_word_tr = set(stopwords.words("turkish"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if len(word) > 2 and word not in stop_word_tr])
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

#################### TF-IDF Özellik Çıkartımı ####################

# TF-IDF vektörleştirme (L2 normalizasyonu ile)
vectorizer = TfidfVectorizer(norm='l2')

X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

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
