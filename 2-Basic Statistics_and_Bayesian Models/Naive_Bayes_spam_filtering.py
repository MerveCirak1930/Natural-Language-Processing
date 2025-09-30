
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# turkish_spam_dataset.csv dosyası, create_tr_dataset_for_spam_filtering.py dosyası kullanılarak bu çalışma için oluşturulmuştur.

# Veri setini yükleme
save_path = "2-Basic Statistics_and_Bayesian Models/turkish_spam_dataset.csv"
data = pd.read_csv(save_path)  # Veri setinin yolunu belirtin

# Özellikler ve etiketler
X = data['text']  # E-posta metinleri
y = data['label']  # Etiketler: 1 = Spam, 0 = Normal

# Eğitim ve test verilerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vektörizasyonu
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Naïve Bayes modelini oluşturma ve eğitme
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


# Test verileri üzerinde tahmin yapma
y_pred = model.predict(X_test_tfidf)


# Sınıflandırma raporu
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))


while True:
    user_email = input("\nTest etmek için bir text girin (Çıkmak için 'q' tuşlayın): ")
    if user_email.lower() == 'q':
        print("Çıkış yapılıyor...")
        break

    user_email_tfidf = vectorizer.transform([user_email])
    prediction = model.predict(user_email_tfidf)[0]

    if prediction == 1:
        print("Bu e-posta **SPAM** olabilir! 🚨")
    else:
        print("Bu e-posta **Normal** görünüyor. ✅")

# CIKTI
# tebrikler-->normal
# fırsatı kaçırmayın-->spam
# çekilişi kazandını-->normal
