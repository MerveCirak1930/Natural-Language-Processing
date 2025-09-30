import os
import pandas as pd


################## Veri Seti Oluşturma ####################

# Kaynak: http://www.kemik.yildiz.edu.tr/veri_kumelerimiz.html
# 	3 sinifa (pozitif, negatif, tarafsiz) ait 105 film kritigi (ham ve arff formatinda)	film_yorumlari.zip klasörü indirilerek 
#   text ve label lar dataframe formatında düzenlenmiştir

# Raw_texts klasör yolunu tanımlayalım
folder_path = "4-Frequency-Based Methods/BoW/raw_texts"  # Raw_texts klasörünün tam yolu

# Klasör isimlerini sınıf etiketleri olarak alalım
class_names = ['pozitif', 'negatif', 'tarafsız']

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

#print(df[30:40])


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


#################### BoW (Bag of Words) Özellik Çıkartımı  ####################

# Metinleri sayısal vektörlere dönüştürme
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

feature_names = vectorizer.get_feature_names_out()
#print("Kelime Kümesi:", feature_names)


#################### Model Eğitimi (Multinomial Naive Bayes-3 sınıf var) ####################

# Naive Bayes, BoW vektörizasyonuyla daha iyi çalışır çünkü BoW, Naive Bayes'in "bağımsızlık varsayımı" ile uyumludur ve 
# bu, modelin kelimelerin sınıflar üzerindeki bağımsız etkilerini hızlı bir şekilde hesaplamasına olanak tanır. 
# Ayrıca BoW, Naive Bayes'in hızını ve basitliğini artırarak büyük metin verileri üzerinde iyi sonuçlar almasını sağlar.

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Veriyi eğitim ve test setlerine ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multinomial Naive Bayes modelini oluşturma ve eğitme
model = MultinomialNB()
model.fit(X_train, y_train)

# Modeli test etme
y_pred = model.predict(X_test)

# Sonuçları değerlendirme
print("Doğruluk: ", accuracy_score(y_test, y_pred))
print(" MNB Sınıflandırma Raporu: \n", classification_report(y_test, y_pred))

# NOT: En iyi sınıflandırma performansa sahip olan Multinominal bayes ile model eğitilmiştir. Diğer modeller kodun alt kısmındadır.


#################### Model Test Etme ####################
# Kullanıcıdan metin girişi al
user_input = input("Lütfen bir film yorumu girin: ")

# Ön işleme adımları
# Kullanıcı girişini temizle
cleaned_input = clean_text(user_input)

# Kullanıcı girişini vektörleştir
vectorized_input = vectorizer.transform([cleaned_input])

# Model ile tahmin yap
predicted_label = model.predict(vectorized_input)[0]

# Sonucu yazdır
print(f"Modelin tahmini: {predicted_label}")

# ÇIKTILAR:
# hiç beğenmedim--> negatif dedi
# fena sayılmaz-->tarafsız
# hayran kaldım-->pozitif

























# #################### Model Eğitimi (LR-3 sınıf var) ####################
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report

# # Lojistik Regresyon modelini oluşturma ve eğitme
# log_reg_model = LogisticRegression(max_iter=1000)
# log_reg_model.fit(X_train, y_train)

# # Modeli test etme
# y_pred_log_reg = log_reg_model.predict(X_test)

# # Sonuçları değerlendirme
# print("Lojistik Regresyon - Doğruluk: ", accuracy_score(y_test, y_pred_log_reg))
# print("Lojistik Regresyon - Sınıflandırma Raporu: \n", classification_report(y_test, y_pred_log_reg))

# #################### Model Eğitimi (RFC-3 sınıf var) ####################
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report

# # Random Forest modelini oluşturma ve eğitme
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)

# # Modeli test etme
# y_pred_rf = rf_model.predict(X_test)

# # Sonuçları değerlendirme
# print("Random Forest - Doğruluk: ", accuracy_score(y_test, y_pred_rf))
# print("Random Forest - Sınıflandırma Raporu: \n", classification_report(y_test, y_pred_rf))


# #################### Model Eğitimi (SVM-3 sınıf var) ####################
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report

# # SVM modelini oluşturma ve eğitme
# svm_model = SVC(kernel='linear')  # Lineer kernel kullanıyoruz
# svm_model.fit(X_train, y_train)

# # Modeli test etme
# y_pred_svm = svm_model.predict(X_test)

# # Sonuçları değerlendirme
# print("SVM - Doğruluk: ", accuracy_score(y_test, y_pred_svm))
# print("SVM - Sınıflandırma Raporu: \n", classification_report(y_test, y_pred_svm))


# #################### Model Eğitimi (GB-3 sınıf var) ####################
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import accuracy_score, classification_report

# # Gradient Boosting modelini oluşturma ve eğitme
# gb_model = GradientBoostingClassifier(random_state=42)
# gb_model.fit(X_train, y_train)

# # Modeli test etme
# y_pred_gb = gb_model.predict(X_test)

# # Sonuçları değerlendirme
# print("Gradient Boosting - Doğruluk: ", accuracy_score(y_test, y_pred_gb))
# print("Gradient Boosting - Sınıflandırma Raporu: \n", classification_report(y_test, y_pred_gb))


