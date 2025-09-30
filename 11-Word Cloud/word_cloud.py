

import re
import zeyrek
from nltk.corpus import stopwords
import nltk
import sys
import os
import contextlib
import logging

logging.getLogger("zeyrek").setLevel(logging.ERROR)

analyzer = zeyrek.MorphAnalyzer()

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def metin_on_isleme(metin_yolu):
    with open(metin_yolu, "r", encoding="utf-8") as dosya:
        metin = dosya.read()

    metin = metin.lower()
    metin = re.sub(r"[^\w\s]", "", metin)
    metin = re.sub(r"\d+", "", metin)
    kelimeler = metin.split()

    stop_words = set(stopwords.words("turkish"))
    kokler = []

    for kelime in kelimeler:
        if kelime in stop_words:
            continue

        with suppress_stdout_stderr():
            analizler = analyzer.analyze(kelime)

        if analizler and isinstance(analizler[0], list) and len(analizler[0]) > 0:
            ilk_cozum = analizler[0][0]
            try:
                kok = ilk_cozum.dictionary_item.lemma
                kokler.append(kok)
            except AttributeError:
                kokler.append(kelime)
        else:
            kokler.append(kelime)

    temiz_metin = " ".join(kokler)
    return temiz_metin

temiz_metin = metin_on_isleme("11-Word Cloud/text.txt")
print('\n', temiz_metin[:300])


from collections import Counter

def kelime_frekansi(metin):
    kelimeler = metin.split()
    frekans = Counter(kelimeler)
    return frekans

# Örnek kullanım:
frekanslar = kelime_frekansi(temiz_metin)

# En sık 10 kelimeyi göster
for kelime, adet in frekanslar.most_common(10):
    print(f"{kelime}: {adet}")


from wordcloud import WordCloud
import matplotlib.pyplot as plt

def wordcloud_olustur(metin):
    wordcloud = WordCloud( 
        width=800,
        height=400,
        background_color='white',
        font_path='C:/Windows/Fonts/arial.ttf'  # Türkçe karakterler için sistemdeki bir font kullanılmalı
    ).generate(metin)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Kelime Bulutu", fontsize=16)

    # Görseli kaydet
    plt.savefig("word_cloud2.png", dpi=300, bbox_inches='tight')

    plt.show()

# Kullanım:
wordcloud_olustur(temiz_metin)
