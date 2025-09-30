import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


################## Veri Seti Oluşturma ####################

# Kaynak: http://www.kemik.yildiz.edu.tr/veri_kumelerimiz.html
# 	4 sinifa (neseli, uzgun, sinirli, karisik) ait 157 blog örnegi (ham ve arff formatinda) ruh_hali.zip klasörü indirilerek 
#   text ve label lar dataframe formatında düzenlenmiştir

# Raw_texts klasör yolunu tanımlayalım
folder_path = "6-Sentiment Analysis/raw_texts"  # Raw_texts klasörünün tam yolu

# Klasör isimlerini sınıf etiketleri olarak alalım
class_names = ['neseli', 'uzgun', 'sinirli','karisik']

# Verileri saklamak için bir liste oluşturalım
data = []

# Her bir klasör için (pozitif, negatif, tarafsız)
for class_name in class_names:
    class_folder = os.path.join(folder_path, class_name)  # Klasör yolu
    
    # Klasördeki her dosyayı oku
    for file_name in os.listdir(class_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(class_folder, file_name)
            
            # Dosyadaki metni oku (encoding'i ISO-8859-9 olarak değiştiriyoruz)
            try:
                with open(file_path, 'r', encoding='ISO-8859-9') as file:
                    text = file.read()
            except UnicodeDecodeError:
                # Eğer hala hata alırsak, latin1 encoding ile deneyelim
                with open(file_path, 'r', encoding='latin1') as file:
                    text = file.read()
                
            # Metni ve etiketi data listesine ekle
            data.append([text, class_name])

# Veriyi DataFrame'e dönüştür
df = pd.DataFrame(data, columns=['text', 'label'])

# DataFrame'in ilk birkaç satırını kontrol edelim
#print(df.head())

#print(df[130:140])



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



#################### TF-IDF Özellik Çıkartımı ####################
from sklearn.feature_extraction.text import TfidfVectorizer

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
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Kullanacağımız modeller
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# Modellerin performansını saklamak için
results = []

for model_name, model in models.items():
    # Modeli eğit
    model.fit(X_train, y_train)
    # Tahmin yap
    y_pred = model.predict(X_test)
    
    # Performans ölçümleri
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append({
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    })
    
    print(f"Model: {model_name}")
    print(classification_report(y_test, y_pred))
    print("-" * 50)

# Sonuçları DataFrame olarak görelim
results_df = pd.DataFrame(results)
print(results_df)



# En iyi performansa sahip sınıflandırma algoritması : Lojistik Regresyon!

# # Lojistik Regresyon için Hiper parametre optimizasyonu ###########################################################
# from sklearn.model_selection import GridSearchCV

# # Logistic Regression için hiperparametre aralıklarını tanımla
# param_grid = {
#     'C': [0.01, 0.1, 1, 10, 100],        # C değeri: 0.01 düşük regularizasyon, 100 yüksek regularizasyon
#     'penalty': ['l2'],                   # L2 regularizasyon kullanalım (standart). İstersen 'l1' de ekleyebilirsin.
#     'solver': ['lbfgs', 'liblinear']      # L2 için uyumlu solverlar
# }

# # Logistic Regression modelini tanımla
# log_reg = LogisticRegression(max_iter=1000)

# # GridSearchCV ile en iyi parametreleri bul
# grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)

# # En iyi parametreleri yazdır
# print("En iyi parametreler:", grid_search.best_params_)

# # En iyi modeli al
# best_log_reg = grid_search.best_estimator_

# # Test verisinde performansını ölç
# y_pred = best_log_reg.predict(X_test)

# print("Optimize edilmiş Logistic Regression performansı:")
# print(classification_report(y_test, y_pred))

#################### Kullanıcıdan Girdi Alarak Test Etme ####################

# Eğitimde kullanılan TF-IDF vektörizer'ı ve Logistic Regression modelini kullanacağız.

# Önce Logistic Regression modelini yeniden eğitelim (veya yukarıda eğittiğin modeli kullanalım)
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train, y_train)

# Kullanıcıdan metin alarak tahmin yap
while True:
    user_input = input("\n\nBir metin girin (çıkmak için 'q' yazın): ")
    if user_input.lower() == 'q':
        print("Program sonlandırıldı.")
        break

    # Kullanıcı metnini temizle
    cleaned_input = clean_text(user_input)

    # TF-IDF vektörize et
    input_tfidf = vectorizer.transform([cleaned_input])

    # Tahmin yap
    prediction = log_reg_model.predict(input_tfidf)
    
    # Tahmin sonucunu yazdır
    print(f"Tahmin edilen ruh hali: {prediction[0]}")
    print("-" * 50)

