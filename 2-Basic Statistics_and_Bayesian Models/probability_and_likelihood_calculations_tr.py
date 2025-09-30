

import nltk
from nltk.util import ngrams
from collections import Counter
import numpy as np
import math

nltk.download('punkt')

# Örnek metin verisi (daha büyük veri setleriyle daha iyi sonuçlar elde edilebilir)
text = "doğal dil işleme alanında dil modelleri önemli bir yer tutar. dil işleme modelleri, metin anlamlandırma ve sınıflandırma için kullanılır."

# 1️⃣ Metni kelimelere ayırma
tokens = nltk.word_tokenize(text.lower())

# 2️⃣ Bigram'ları oluşturma
bigrams = list(ngrams(tokens, 2))

# 3️⃣ Frekansları hesaplama
bigram_freq = Counter(bigrams)  # Bigram frekansları
unigram_freq = Counter(tokens)   # Tekil kelime frekansları

# 4️⃣ Olasılık Hesaplama (Probability)
def bigram_probability(word1, word2):
    """P(word2 | word1) = Count(word1, word2) / Count(word1)"""
    bigram_count = bigram_freq.get((word1, word2), 0)
    unigram_count = unigram_freq.get(word1, 1)  # Eğer kelime yoksa 1 alarak sıfır hatasını önlüyoruz
    return bigram_count / unigram_count

# Örnek Probability Hesaplamaları
print("\nBigram Olasılıkları (P(word2 | word1)):")
for bigram in bigram_freq.keys():
    prob = bigram_probability(bigram[0], bigram[1])
    print(f"P({bigram[1]} | {bigram[0]}) = {prob:.4f}")

# 5️⃣ Likelihood (Olasılık Değeri) Hesaplama
def sentence_likelihood(sentence):
    """Bir cümlenin likelihood değerini hesaplar"""
    sentence_tokens = nltk.word_tokenize(sentence.lower())
    sentence_bigrams = list(ngrams(sentence_tokens, 2))
    
    likelihood = 1
    for bigram in sentence_bigrams:
        likelihood *= bigram_probability(bigram[0], bigram[1])  # Çarpılarak devam ediyoruz

    return likelihood

# Örnek Cümlelerin Likelihood Değerleri
sentence1 = "dil işleme modelleri önemli"
sentence2 = "modelleri dil işleme"   

likelihood1 = sentence_likelihood(sentence1)
likelihood2 = sentence_likelihood(sentence2)

print("\nCümlelerin Likelihood Değerleri:")
print(f"L('{sentence1}') = {likelihood1:.8f}")
print(f"L('{sentence2}') = {likelihood2:.8f}")



# # Cıktıları yorumlama
# Olasılık (Probability): Örneğin, "dil" kelimesinden sonra "işleme" kelimesinin gelme olasılığı %67 olarak hesaplandı.
# Likelihood (Olabilirlik Değeri): "dil işleme modelleri önemli" cümlesinin, modeli oluşturan metin içinde ne kadar olası olduğunu gösteriyor.