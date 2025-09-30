

from sklearn.feature_extraction.text import CountVectorizer

# ornek metin
documents = [
    "Bu bir örnek metindir",
    "Bu örnek metin doğal dil işlemeyi gösterir."]

# unigram, bi gram ve trigram -> CountVectorizer

vectorizer_unigram = CountVectorizer(ngram_range=(1,1))
vectorizer_bigram = CountVectorizer(ngram_range=(2,2))
vectorizer_trigram = CountVectorizer(ngram_range=(3,3))

# unigram
X_unigram = vectorizer_unigram.fit_transform(documents)
unigram_features = vectorizer_unigram.get_feature_names_out()

# bigram
X_bigram = vectorizer_bigram.fit_transform(documents)
bigram_features = vectorizer_bigram.get_feature_names_out()

# trigram
X_trigram= vectorizer_trigram.fit_transform(documents)
trigram_features = vectorizer_trigram.get_feature_names_out()

print("unigram_features: ",unigram_features)
print("bigram_features: ",bigram_features)
print("trigram_features: ",trigram_features)