

# Kodun Amacı: 

# Kullanıcı tarafından seçilen kelime gömme modeli (Word2Vec, GloVe, FastText) ile:
# Kelime çiftlerinin model tarafından tahmin edilen benzerliğini (cosine similarity)
# İnsan anotasyonları tarafından verilmiş olan gerçek benzerlik (Sim) ve ilişkililik (Rel) skorlarıyla karşılaştırmak.


# Bu kod ne yapıyor?

# Anlamver veri setindeki kelime çiftlerinin her biri, seçilen kelime gömme modeli tarafından işlenir, 
# kelime gömme modelinde bulunan kelime çiftleri için benzerlik skorları(cosine similarity => -1 ile 1 arasında) hesaplanır
# bu skorlar ile anlamver veri setindeki insan/gerçek anotasyonları tarafından verilmiş olan gerçek benzerlik (Sim) ve ilişkililik (Rel) skorları
# Spearman korelasyonu kullanarak karşılaştırılır.


# Neden Spearman korelasyonu kullanılıyor?

# 1. Ölçek Uyuşmazlığı ve Dağılımsal Varsayımlar
# 2. Model ve İnsan Algısı Arasındaki Sıralama Tutarlılığını Ölçme

# Spearman korelasyonu, NLP’deki embedding modellerinin çıktıları ile insan anotasyonlarının sıralama düzeyinde ne kadar benzeştiğini ölçer. 
# Bu sayede, mutlak skorların karşılaştırılamadığı durumlarda bile 
# semantik benzerlikteki göreli başarıyı güvenilir bir şekilde değerlendirmek mümkün olur.



# Gerekli Kütüphanelerin Yüklenmesi

import pandas as pd                         # Veri işlemek için kullanılır, CSV dosyasını okur ve DataFrame oluşturur.
from gensim.models import KeyedVectors      # Word2Vec, GloVe ve FastText modellerini yüklemek ve bu modellerle kelime benzerliklerini hesaplamak için kullanılır.
from scipy.stats import spearmanr           # Spearman korelasyonunu hesaplama


# ----------------------------------------------
# 1. Model (kelime gömme modeli) seçimi - Kullanıcı etkileşimi ile
# ----------------------------------------------
model_codes = {
    "w": "word2vec",
    "g": "glove",
    "f": "fasttext"
}

while True:
    user_input = input("Model seç (w: Word2Vec, g: GloVe, f: FastText): ").strip().lower()
    if user_input in model_codes:
        model_name = model_codes[user_input]
        break
    else:
        print("Geçersiz seçim. Lütfen 'w', 'g' veya 'f' girin.")

# 🔹 GloVe (glove.42B.300d.zip)

# Veri Kümesi: Common Crawl (42 milyar kelime)
# Kelime Sayısı: Yaklaşık 1.9 milyon
# Büyük/küçük harf ayrımı: Uncased (yani "Apple" ve "apple" aynıdır)
# Vektör Boyutu: 300
# Dosya Boyutu: ~1.75 GB
# Bu model, geniş çapta toplanan web verisi olan Common Crawl üzerinde eğitilmiş. GloVe modelleri, kelimeler arası istatistiksel ilişkileri öğrenmekte başarılıdır.

# 🔹 Word2Vec (GoogleNews-vectors-negative300.bin.gz)
# Veri Kümesi: Google News (yaklaşık 100 milyar kelime)
# Kelime ve İfade Sayısı: ~3 milyon
# Vektör Boyutu: 300
# Dosya Boyutu: ~1.5 GB
# Google tarafından yayınlanan bu model, İngilizce haber metinleri üzerinde eğitilmiş. Hem kelimeler hem de bazı çok sözcüklü ifadeler (multi-word expressions) içerir.

# 🔹 FastText (wiki-news-300d-1M-subword.vec.zip)
# Veri Kümesi:
#     Wikipedia 2017
#     UMBC webbase corpus
#     statmt.org haber veri seti
#     (toplam: 16 milyar kelime)
# Kelime Sayısı: 1 milyon
# Vektör Boyutu: 300
# FastText, kelime içi alt-birimleri (subword units) de kullandığı için nadir veya görülmemiş kelimelere bile vektör atayabilir. 
# Bu da onu özellikle düşük kaynaklı veya türetilmiş kelimelerde avantajlı kılar.

# ----------------------------------------------
# 2. Model yolları ve binary ayarları
# ----------------------------------------------

# ### glove.42B.300d.txt dosyasını dönüştürme (sadece 1 kez çalıştır)
# from gensim.scripts.glove2word2vec import glove2word2vec

# # Orijinal GloVe dosyan
# glove_input_file = "5-Word Embeddings/models/glove.42B.300d.txt"

# # Oluşturulacak Word2Vec uyumlu dosya (bu yeni dosya olacak)
# word2vec_output_file = "5-Word Embeddings/models/glove.42B.300d.word2vec.txt"

# # Dönüştürme işlemi
# glove2word2vec(glove_input_file, word2vec_output_file)

# print("Dönüştürme tamamlandı.")
# ### glove.42B.300d.txt dosyasını dönüştürme (sadece 1 kez çalıştır) sonu


model_paths = {
    'word2vec': "5-Word Embeddings/models/GoogleNews-vectors-negative300.bin",
    'glove': "5-Word Embeddings/models/glove.42B.300d.word2vec.txt",  # <-- dönüştürülmüş dosya!
    'fasttext': "5-Word Embeddings/models/wiki-news-300d-1M-subword.vec"
}

model_binary = {
    'word2vec': True,
    'glove': False,
    'fasttext': False
}

# ----------------------------------------------
# 3. Seçilen kelime gömme modelinin yüklenmesi
# ----------------------------------------------
print(f"\n{model_name} modeli yükleniyor...")
model = KeyedVectors.load_word2vec_format(model_paths[model_name], binary=model_binary[model_name])
print(f"{model_name} modeli başarıyla yüklendi.\n")


# ----------------------------------------------
# 4. Veri setinin yüklenmesi
# ----------------------------------------------

# # Veri, anlamver-final.csv dosyasından pandas ile yüklenir.
# df = pd.read_csv("5-Word Embeddings/models/dataset/anlamver-final.csv", delimiter=";", encoding="ISO-8859-9", decimal=",")
# #print(df.head()) 

# # Dosyayı düzenle ve yeni bir adla kaydet (örneğin '_duzenlenmis' eki ile)
# df.to_csv("5-Word Embeddings/dataset/anlamver_final_duzenlenmis.csv", 
#           index=False, 
#           encoding="utf-8",  # veya "ISO-8859-9"
#           sep=";")
       

# Veri, anlamver-final.csv dosyasından pandas ile yüklenir.
df = pd.read_csv("5-Word Embeddings/dataset/anlamver-final.csv", delimiter=";", encoding="ISO-8859-9", decimal=",")
#print(df.head()) 


# ----------------------------------------------
# 5. Bos listelerin oluşturulması ve skorların hesaplanması
# ----------------------------------------------


# Her kelime gömme modelinin tahmin ettiği benzerlik skorları ve anlamver veri setindeki gerçek anotasyonlar için boş listeler oluşturulur.

actual_sim_scores, actual_rel_scores = [], []   # anlamver veri setindeki benzerlik ve ilişiklilik skorlarının listelenmesi
predicted_sim_scores = []                       # kelime gömme modelinin tahmin ettiği benzerlik skorları
w1_list, w2_list = [], []                       # Anlamver veri setindeki kelime çiftleri
in_vocab_pairs = []                             # Modelde bulunan kelime çiftlerinin listelenmesi
missing = 0                                     # Modelde bulunmayan kelime çiftlerinin sayısı


# Anlamver veri setindeki kelime çiftlerinin seçilen kelime gömme modeli tarafından işlenmesi
for _, row in df.iterrows():
    word1, word2 = row["W1"], row["W2"]
    
    if word1 in model and word2 in model:
        sim_score = model.similarity(word1, word2) # kelime gömme modelinin, kelime çifti arasındaki cosinus benzerliğini hesaplaması
        predicted_sim_scores.append(sim_score)     # benzerlik skorlarının listeye eklenmesi
        actual_sim_scores.append(row["Sim"])       # anlamver veri setindeki kelime çiftleri arsındaki benzerlik skorunun listeye eklenmesi
        actual_rel_scores.append(row["Rel"])       # anlamver veri setindeki kelime çiftleri arsındaki ilişiklilik skorunun listeye eklenmesi
        w1_list.append(word1)                      # anlamver veri setindeki kelime çiftlerinden 1.sinin listeye eklenmesi (kelime gömme modelinde varsa)
        w2_list.append(word2)                      # anlamver veri setindeki kelime çiftlerinden 2.sinin listeye eklenmesi (kelime gömme modelinde varsa)
        in_vocab_pairs.append((word1, word2))      # kelime gömme modelinde bulunan kelime çiftlerinin listeye eklenmesi
    else:
        missing += 1

# ----------------------------------------------
# 6. DataFrame oluştur (kelime çifti, model tahmini ve gerçek skorların bir araya getirilmesi)
# ----------------------------------------------
df_results = pd.DataFrame({
    "W1": w1_list,
    "W2": w2_list,
    "Actual_Sim": actual_sim_scores,
    "Actual_Rel": actual_rel_scores,
    f"{model_name}_Sim": predicted_sim_scores,
})


# ----------------------------------------------
# 7. Çıktıları yazdır
# ----------------------------------------------
print(f"\nToplam: {len(in_vocab_pairs)} kelime çifti bulundu.")
print(f"\n{model_name} modelinde bulunan kelime çiftleri:")
# for pair in in_vocab_pairs:
#     print(pair)

print(f"\nModelde olmayan kelime çiftleri sayısı: {missing}")

print(f"\n\nModelde bulunan kelime çiftleri - Benzerlik ve İlişiklilik değerleri\n")
print(df_results[["W1", "W2", "Actual_Sim", "Actual_Rel"]].to_string(index=False))
#print(df_results[["W1", "W2", "Actual_Sim", "Actual_Rel", "{model_name}_Sim"]].to_string(index=False))

# ----------------------------------------------
# 8. Spearman Korelasyon Hesapla
# ----------------------------------------------
print('\nSpearman korelasyonu ile kelime gömme modeli tahmini ve insan anatasyonlarının karsılastırılması:')

correlation_sim, _ = spearmanr(df_results["Actual_Sim"], df_results[f"{model_name}_Sim"])
print(f"\n Comparison of similarity scores between {model_name} and (Sim) with Spearman Correlation:", correlation_sim)

correlation_rel, _ = spearmanr(df_results["Actual_Rel"], df_results[f"{model_name}_Sim"])
print(f"\n Comparison of similarity scores between {model_name} and (Rel) with Spearman Correlation:", correlation_rel)


# ----------------------------------------------
# 9. Çıktıların yorumlanması:
# ----------------------------------------------


# Word2Vec modelinin çıktısına bakıldığında:
    # Kelime Çiftleri: Yalnızca 2 kelime çifti üzerinde işlem yapılabilmiş, bu da modelin sınırlı bir veri setiyle çalıştığını gösteriyor.
    # Benzerlik Skorları: 2 kelime çifti için benzerlik değerleri verilmiş ancak bu çok küçük bir örnekle sınırlı olduğu için güvenilirlik düşük.
    # Spearman Korelasyon: Benzerlik skorları için mükemmel bir korelasyon (1.0) gözlemlenmiş, 
    #   ancak sadece 2 kelime çifti üzerinden yapılan değerlendirme, genellenebilir bir sonuç elde etmeyi zorlaştırıyor.
    # İlişkililik Skorları: Yetersiz veri yüzünden ilişkililik skorları hesaplanamıyor (nan), 
    #   bu da modelin çok sınırlı bir veri kümesiyle çalışmasından kaynaklanıyor.
    # Özetle, modelin değerlendirmesi küçük bir veri setine dayandığı için sonuçlar genellenebilir değil.


# GloVe modelinin Spearman korelasyon değerleri düşük çıkmış:
    # Benzerlik (Sim) Korelasyonu: 0.076 – Model, insan anotasyonlarıyla benzerlik sıralamalarında çok uyumlu değil.
    # İlişkililik (Rel) Korelasyonu: 0.188 – Model, ilişkilik sıralamalarında biraz daha uyumlu, ancak hala düşük.
    # Özetle, GloVe modeli, insan benzerlik ve ilişkilik sıralamalarını tahmin etmede yeterince başarılı olmamış.


# FastText modelinin çıktısına göre:
    # Benzerlik (Sim) Skorları: Modelin tahmin ettiği sıralamalar, insan anotasyonlarıyla ters yönde, yani negatif korelasyon (-0.21).
    # İlişkililik (Rel) Skorları: Modelin tahminleri ile insan anotasyonları arasında zayıf bir pozitif ilişki var (0.21).
    # Özetle, FastText modelinin benzerlik tahminleri zayıf ve yanlış sıralama yapıyor, ancak ilişkililik tahminleri kısmen doğru.