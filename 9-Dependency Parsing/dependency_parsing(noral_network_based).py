
# Stanza aracı ve Biaffine Scoring algoritması ile Nöral ağ tabanlı bağımlılık çözümlemesi: 

# Nöral ağ tabanlı bağımlılık çözümlemesinde, Stanza aracı;
# Türkçe metinlerde kelimeler arasındaki sözdizimsel bağımlılıkları derin öğrenme tabanlı Biaffine Scoring algoritmasıyla tespit ederek etkili çözümleme yapar.

import stanza
import networkx as nx
import matplotlib.pyplot as plt

# İlk kullanımda modelin indirilmesi gerekir
stanza.download('tr')

# NLP pipeline'ı oluştur
nlp = stanza.Pipeline(lang='tr', processors='tokenize,mwt,pos,lemma,depparse')

# Kullanıcıdan cümle al
user_input = input("Lütfen bir Türkçe cümle girin (ya da Enter'a basarak örnek cümleyi kullanın): ")
text = user_input.strip() if user_input.strip() else "Kedi pencereden dışarıya dikkatle bakıyor."

# Cümleyi analiz et
doc = nlp(text)

# İngilizce bağımlılık etiketlerine karşılık gelen Türkçe açıklamalar
deprel_dict = {
    "nsubj": "özne",
    "obj": "nesne",
    "iobj": "dolaylı tümleç",
    "amod": "sıfat tümleci",
    "advmod": "zarf tümleci",
    "obl": "dolaylı tümleç (yer/zaman)",
    "nmod": "isim tamlaması",
    "det": "belirleyici (belirteç)",
    "case": "ilgeç (edat vs.)",
    "cc": "bağlaç",
    "conj": "bağlı cümle/öğe",
    "mark": "bağlaç (yan cümle)",
    "cop": "yüklem fiili (olmak)",
    "compound": "bileşik kelime",
    "xcomp": "açık tümleç (fiil)",
    "aux": "yardımcı fiil",
    "root": "kök",
    "punct": "noktalama işareti",
    "acl": "sıfat cümleciği",
    "advcl": "zarf cümleciği",
    "appos": "açıklayıcı öge",
    "csubj": "yan cümle öznesi",
    "ccomp": "yan cümle nesnesi",
}

# Açıklamalı analiz çıktısı
print(f"\nBağımlılık çözümlemesi sonucu ({text}):\n")

for sentence in doc.sentences:
    for word in sentence.words:
        head_text = "ROOT" if word.head == 0 else sentence.words[word.head - 1].text
        deprel = word.deprel
        turkish_desc = deprel_dict.get(deprel, "bilinmiyor")  # Etiket sözlükte yoksa "bilinmiyor" yaz
        print(f"""
Kelime        : {word.text}
Kelimenin Pozisyonu      : {word.id}
Kelimenin Bağlı Olduğu Kelime  : {head_text} (id={word.head})
İlişki Türü (Bağımlılık etiketi)   : {deprel} ({turkish_desc})
-------------------------------
""")

# Bağımlılık ağacının görselleştirilmesi
G = nx.DiGraph()

for sentence in doc.sentences:
    for word in sentence.words:
        head = "ROOT" if word.head == 0 else sentence.words[word.head - 1].text
        G.add_edge(head, word.text)

plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=12)
plt.title("Türkçe Bağımlılık Grafiği")
plt.show()



