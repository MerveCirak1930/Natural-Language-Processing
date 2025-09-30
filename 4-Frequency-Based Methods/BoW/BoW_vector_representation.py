

from sklearn.feature_extraction.text import CountVectorizer

# Metin
documents = [
    "kedi evde",
    "kedi bahçede"
]

# bow
vectorizer  = CountVectorizer()

# metni sayısal vektöre dönüstürme (metne vektorizer uygulama)
X = vectorizer.fit_transform(documents)

# Kelime kümesi 
print("Kelime kümesi:",vectorizer.get_feature_names_out()) #['bahçede' 'evde' 'kedi']

print("Vektör temsili:",X.toarray()) # Vektör temsili: [[0 1 1]
                                    #                  [1 0 1]]