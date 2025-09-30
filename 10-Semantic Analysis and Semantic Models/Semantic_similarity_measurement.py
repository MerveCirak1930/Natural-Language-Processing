
# Uygulama: Anlamsal Benzerlik Ölçme Aracı

from sentence_transformers import SentenceTransformer, util

# Modeli yükle
# Bu model, çok dilli (Türkçe dahil) olarak cümle benzerliği için özel olarak eğitilmiş bir Sentence-BERT modelidir.
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


# Kullanıcıdan cümleleri al
cumle1 = input("1. Cümleyi girin: ")
cumle2 = input("2. Cümleyi girin: ")

# Cümleleri embed et
embedding1 = model.encode(cumle1, convert_to_tensor=True)
embedding2 = model.encode(cumle2, convert_to_tensor=True)

# Kosinüs benzerliğini hesapla
benzerlik = util.cos_sim(embedding1, embedding2)

print(f"\nCümleler Arası Anlamsal Benzerlik: {benzerlik.item():.4f} (0.0 - 1.0 arası)\n\n")

# Kedi süt içiyor
# Köpek mama yedi
# Taksi trafiğe çıktı.
# Otobüs durağa yanaştı
# okula yarın gelecek
