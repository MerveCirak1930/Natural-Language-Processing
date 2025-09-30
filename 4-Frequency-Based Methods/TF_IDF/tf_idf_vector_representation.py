

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

# ornek belge
documents = [
    "Kedi çok tatlı bir hayvandır",
    "Kedi ve köpekler çok tatlı hayvanlardır",
    "Arılar bal üretirler"
    ]

tfidf_vectorizer = TfidfVectorizer()

# metinler -> sayisal
X = tfidf_vectorizer.fit_transform(documents)

# kelime kumesi
feature_names = tfidf_vectorizer.get_feature_names_out()

print("TF-IDF Vektor temsilleri:")
vektor_temsili = X.toarray()
print(vektor_temsili)

df_tfidf = pd.DataFrame(vektor_temsili, columns = feature_names)
print(df_tfidf)

kedi_tfidf = df_tfidf["kedi"]
kedi_mean_tfidf = np.mean(kedi_tfidf)