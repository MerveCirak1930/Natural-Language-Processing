
# Python ile basit bir N-gram modeli 

# Adımlar:
# Metin verisini temizleme.
# N-gramları oluşturma.
# N-gramların sıklığını hesaplama.

 
import nltk
from nltk.util import ngrams
from collections import Counter
from nltk.tokenize import word_tokenize

# NLTK tokenizasyonu için gerekli veri setini indir
nltk.download("punkt")

# Varsayılan metin
default_text1 = "Doğal dil işleme, dil işleme teknikleri ile çalışır. Dil işleme, metinleri anlamlandırmak için kullanılır. Yapay zeka ve dil işleme birlikte çalışır."

default_text2 = "Dil modelleri, doğal dil işleme alanında önemli bir rol oynar. Bu modeller, metinlerin anlaşılmasını ve işlenmesini sağlar."

print("Kod başlatıldı!")  # İlk satıra ekle

while True:
    # Kullanıcıdan varsayılan metni kullanıp kullanmayacağını sorma
    use_default = input("\nVarsayılan metinlerden kullanmak ister misiniz? e(evet), h(hayır), 'q'(cıkıs) yazın: ").strip().lower()

    # Çıkış komutu
    if use_default == "q":
        print("Programdan çıkılıyor...")
        break

    # Eğer "evet" denirse varsayılan metni kullan, "hayır" denirse kullanıcıdan metin al
    elif use_default == "e":
        secim = input("Lütfen analiz etmek istediğiniz metni secin: 1 ya da 2 :").strip()

        if secim == "1":
            text = default_text1
        elif secim == "2":
            text = default_text2
        else:
            print("Geçersiz giriş! Lütfen '1' veya '2' yazın.")
            continue  # Geçersiz girişte başa dön

    elif use_default == "h":
        text = input("Lütfen analiz etmek istediğiniz metni girin: ").strip()
    
    else:
        print("Geçersiz giriş! Lütfen 'e', 'h' veya 'q' yazın.")
        continue  # Geçersiz girişte başa dön

    # 1. Metni kelimelere ayırma
    tokens = word_tokenize(text.lower())  # Küçük harfe çevirerek token'ları oluştur

    # 2. N-gram'ları oluşturma (N=2, yani bigram)
    n = 2
    bigrams = ngrams(tokens, n)

    # 3. N-gram'ların sıklığını hesaplama
    bigram_freq = Counter(bigrams)

    # 4. Sonuçları yazdırma
    print("\nBigram Frekansları:")
    for bigram, freq in bigram_freq.items():
        print(f"{bigram}: {freq}")

