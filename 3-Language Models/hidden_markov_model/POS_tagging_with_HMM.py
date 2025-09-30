

# Bu çalışmada, Hidden Markov Model (HMM) kullanarak Türkçe bir metin üzerinde kelime etiketleme (Part-of-Speech Tagging) yapılmıştır.
# HMM, dilin gizli yapısını ve kelimeler arasındaki etkileşimleri modellemek için etkili bir yaklaşımdır. 
# Bu süreçte, metinler üzerinde dilbilgisel etiketlerin (örneğin, NOUN, VERB, ADJ) tahmin edilmesi amaçlanmaktadır.

# ---------------------------
#  1. Veri Kümesini Yükleme
# ---------------------------

# Veri kümesi, conllu formatındaki Türkçe bir dil verisi seti olan tr_imst-ud-dev.conllu dosyasından alınmıştır. 
# Bu dosya, her kelime için dilbilgisel etiketlerin bulunduğu bir yapıya sahiptir. 
# Etiketler, dildeki kelimelerin hangi türden olduğunu belirler. 


from conllu import parse_incr

# Veri dosyasının yolunu belirt
file_path = "3-Language Models/tr_imst-ud-dev.conllu"

sentences = []

# Dosyayı satır satır oku ve en az 3 kelimelik cümleleri al
with open(file_path, "r", encoding="utf-8") as f:
    for sentence in parse_incr(f):
        word_tag_pairs = [(token['form'], token['upostag']) for token in sentence if token['upostag'] is not None]
        if len(word_tag_pairs) >= 3:
            sentences.append(word_tag_pairs)

# İlk 5 örnek cümleyi yazdır
# for s in sentences[:5]:
#     print(s)

# -----------------------------------------------
#  2. Sayımlar: Başlangıç, Geçiş ve Emisyon
# -----------------------------------------------

from collections import defaultdict, Counter

start_counts = Counter()
transition_counts = defaultdict(Counter)
emission_counts = defaultdict(Counter)
tag_counts = Counter()

# Etiket sayımlarını yap
for sentence in sentences:
    prev_tag = None
    for idx, (word, tag) in enumerate(sentence):
        tag_counts[tag] += 1
        emission_counts[tag][word] += 1

        if idx == 0:
            start_counts[tag] += 1  # İlk kelimenin etiketi başlangıç etiketi sayılır
        else:
            transition_counts[prev_tag][tag] += 1  # Önceki etiketten şimdiki etikete geçiş

        prev_tag = tag

# --------------------------------------
#  3. Olasılıkların Hesaplanması
# --------------------------------------

# Başlangıç olasılıkları (start_prob): Bir cümlenin hangi etiketle başladığına dair olasılıkları hesapladık.
total_starts = sum(start_counts.values())
start_prob = {tag: count / total_starts for tag, count in start_counts.items()}

# Geçiş olasılıkları (transition_prob): Bir etiketin bir başka etikete geçme olasılıklarını hesapladık.
transition_prob = {}
for prev_tag in transition_counts:
    total = sum(transition_counts[prev_tag].values())
    transition_prob[prev_tag] = {tag: count / total for tag, count in transition_counts[prev_tag].items()}

# Emisyon olasılıkları (emission_prob): Her etiketin, hangi kelimelerle ilişkili olduğunu ve kelimenin etiketine dair olasılıkları hesapladık.
emission_prob = {}
for tag in emission_counts:
    total = sum(emission_counts[tag].values())
    emission_prob[tag] = {word: count / total for word, count in emission_counts[tag].items()}

# -------------------------------
#  4. Olasılıkları Yazdır
# -------------------------------

# print("📌 Başlangıç Olasılıkları (start_prob):")
# for tag, prob in start_prob.items():
#     print(f"{tag}: {prob:.4f}")

# print("\n📌 Geçiş Olasılıkları (transition_prob) örnek:")
# for prev_tag in list(transition_prob.keys())[:3]:
#     print(f"{prev_tag} → {transition_prob[prev_tag]}")

# print("\n📌 Emisyon Olasılıkları (emission_prob) örnek:")
# for tag in list(emission_prob.keys())[:3]:
#     print(f"{tag} → {list(emission_prob[tag].items())[:5]}")

# -------------------------------------
#  5. Viterbi Algoritması Tanımlama
# -------------------------------------


# Viterbi Algoritması, sırasıyla bir cümledeki her kelime için en yüksek olasılığa sahip etiketlerin seçilmesini sağlar.
# Bu algoritma, başlangıç olasılıkları, geçiş olasılıkları ve emisyon olasılıklarını kullanarak en iyi etiket sırasını bulmaya çalışır.
# Viterbi Tablosu (V) ve Backpointer (geri izleme işlevi) kullanarak, her kelimenin en olası etiketini ve etiketler arası geçişleri hesapladık.


import numpy as np

def viterbi(words, start_prob, transition_prob, emission_prob, tag_list):
    V = [{}]                # Viterbi tablosu
    backpointer = [{}]      # En iyi yolun tutulduğu tablo

    # İlk kelime için başlatma adımı
    for tag in tag_list:
        emission = emission_prob.get(tag, {}).get(words[0], 1e-6)
        V[0][tag] = start_prob.get(tag, 1e-6) * emission
        backpointer[0][tag] = None

    # Devam eden kelimeler için olasılık hesabı
    for t in range(1, len(words)):
        V.append({})
        backpointer.append({})
        for tag in tag_list:
            max_prob = 0
            best_prev_tag = None
            emission = emission_prob.get(tag, {}).get(words[t], 1e-6)
            for prev_tag in tag_list:
                prob = V[t-1][prev_tag] * transition_prob.get(prev_tag, {}).get(tag, 1e-6) * emission
                if prob > max_prob:
                    max_prob = prob
                    best_prev_tag = prev_tag
            V[t][tag] = max_prob
            backpointer[t][tag] = best_prev_tag

    # En yüksek olasılığa sahip son etiketi bul
    last_tag = max(V[-1], key=V[-1].get)
    best_path = [last_tag]

    # Geriye doğru en iyi yolu takip ederek etiketleri bul
    for t in range(len(words)-1, 0, -1):
        best_path.insert(0, backpointer[t][best_path[0]])

    return best_path

# ---------------------------------------------
#  6. Kullanıcıdan Cümle Alma / Varsayılan
# ---------------------------------------------

tag_list = list(tag_counts.keys())

secim = input("\nTest cümlesini kendin girmek ister misin? (E/h): ").strip().lower()

if secim == 'e':
    cumle_str = input("Lütfen test cümlesini gir (kelimeler arasında boşluk olmalı): ")
    test_sentence = cumle_str.strip().split()
else:
    test_sentence = ["Ben", "kitap", "okurum"]

# --------------------------------
#  7. Viterbi ile Tahmin Et
# --------------------------------

predicted_tags = viterbi(test_sentence, start_prob, transition_prob, emission_prob, tag_list)

# Sonuçları yazdır
print("\n✅ Test Cümlesi Etiketleme Sonucu:")
for word, tag in zip(test_sentence, predicted_tags):
    print(f"{word} → {tag}")
