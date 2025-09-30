

#  Amaç: Verilen bir cümlede geçen çok anlamlı bir kelimenin, WordNet sözlüğündeki olası anlam tanımları (gloss) ile karşılaştırılıp, 
# en uygun anlamın otomatik olarak belirlenmesi.

#  Yöntem:
# 1- Cümledeki kelimenin bağlamsal embedding’i alınır (BERT ile).
# 2- WordNet'ten kelimenin anlamları alınır.
# 3- Her gloss cümlesi için de BERT embedding’i çıkarılır.
# 4- Cümle ile gloss arasındaki cosine benzerliği hesaplanır.
# 5- En yüksek benzerliğe sahip gloss, kelimenin anlamı olarak seçilir.

# import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')


import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Türkçe BERT modeli (cased versiyonu)
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, output_hidden_states=True)
model.eval()

# Türkçe cümle ve hedef kelime
sentence = "Sabah erkenden denizde yüz."
#sentence = "Yüzüne bir şey çarptı."

target_word = "yüz"

# Hedef kelime embedding'ini çıkar
def get_word_embedding(text, target):
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    token_indices = [i for i, tok in enumerate(tokens) if target in tok]

    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states

    token_embeddings = torch.stack(hidden_states[-4:]).mean(dim=0)
    word_embedding = token_embeddings[0, token_indices].mean(dim=0)
    return word_embedding


# WordNet yerine Türkçe anlam kaynağı gerekir
# → Türkçe için WordNet benzeri KeNet, OpenWordNet-TR gibi kaynaklar kullanılabilir ama doğrudan nltk.corpus.wordnet desteği yoktur.
# Yine de bu örnekte gloss yerine önceden hazırlanmış anlam tanımlarını manuel olarak tanımlayabiliriz.

# Türkçe "yüz" kelimesinin olası anlam gloss'ları (manuel tanımlar)
senses = [
    ("isim-yuz", "İnsan vücudunun ön kısmında bulunan, göz, burun, ağız gibi organları içeren bölüm."),
    ("fiil-yuz", "Su içinde kollar ve bacakların hareketiyle ilerlemek.")
]

# Cümledeki hedef kelimenin bağlamsal embedding’i
context_embedding = get_word_embedding(sentence, target_word)

# Her anlam (sense) için gloss (tanım) embedding'i çıkar ve karşılaştır
sense_similarities = []
for name, gloss in senses:
    gloss_embedding = get_word_embedding(gloss, target_word if target_word in gloss else gloss.split()[0])
    
    sim = cosine_similarity(
        context_embedding.unsqueeze(0).numpy(),
        gloss_embedding.unsqueeze(0).numpy()
    )[0][0]

    sense_similarities.append((name, gloss, sim))

# En uygun anlamı bul
best_sense = max(sense_similarities, key=lambda x: x[2])

# Sonuçları yazdır
print(f"\nCümle: {sentence}")
print(f"Hedef kelime: {target_word}")
print("\nOlası anlamlar:")
for name, gloss, sim in sense_similarities:
    print(f"  - {name:15s} | {gloss:60s} | Benzerlik: {sim:.4f}")

print(f"\nTahmin edilen en uygun anlam: {best_sense[0]}")
print(f"Tanım: {best_sense[1]}\n\n")
