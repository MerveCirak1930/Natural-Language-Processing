import os
import numpy as np
import pandas as pd


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, MeanShift, SpectralClustering, AgglomerativeClustering
from sklearn.decomposition import NMF
from gensim import corpora, models
# from bertopic import BERTopic
# from sentence_transformers import SentenceTransformer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt


################## Veri Seti Oluşturma ####################

# Kaynak: http://www.kemik.yildiz.edu.tr/veri_kumelerimiz.html
# 	5 sinifa ait 230'ar adet haber metni (ham ve arff formatinda)	1150haber.zip klasörü indirilerek 
#   text ve label lar dataframe formatında düzenlenmiştir

folder_path = "7-Clustering and Topic Modeling/raw_texts"

class_names = ['ekonomi', 'magazin', 'saglik','siyasi','spor']
data = []

for class_name in class_names:
    class_folder = os.path.join(folder_path, class_name)
    for file_name in os.listdir(class_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(class_folder, file_name)
            try:
                with open(file_path, 'r', encoding='ISO-8859-9') as file:
                    text = file.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin1') as file:
                    text = file.read()
            data.append([text, class_name])

df = pd.DataFrame(data, columns=['text', 'label'])

#################### Veri Temizleme ####################
import re

# Veri temizleme fonksiyonu
def clean_text(text):

    # Küçük harfe dönüştürme
    text = text.lower()

    # Sayıları kaldırma
    text = re.sub(r'\d+', '', text)

    # Noktalama işaretlerini kaldırma
    text = re.sub(r'[^\w\s]', '', text)

    # Kısa kelimeleri (2 karakterden az) kaldırma
    text = ' '.join([word for word in text.split() if len(word) > 2])

    # Stop-word'leri temizle
    import nltk
    from nltk.corpus import stopwords

    # Eğer stopwords veri seti eksikse indir
    #nltk.download("stopwords")

    # Türkçe stop-words listesini yükle
    stop_word_tr = set(stopwords.words("turkish"))
    #print(stop_word_tr)
    text = [kelime for kelime in text.split() if kelime not in stop_word_tr]

    return ' '.join(text)

# Metinleri temizle
df['cleaned_text'] = df['text'].apply(clean_text)

# Temizlenmiş metnin ilk birkaç satırına bakalım
#print('temizlenmiş metin:\n',df['cleaned_text'].head())


#################### TF-IDF Özellik Çıkartımı ####################

# TF-IDF vektörleştirme (L2 normalizasyonu ile)
#vectorizer = TfidfVectorizer(norm='l2')

# TF-IDF Parametrelerini Optimize Etme
vectorizer = TfidfVectorizer(norm='l2', ngram_range=(1,2), min_df=2)

X = vectorizer.fit_transform(df['cleaned_text'])
# y = df['label']



##################### Clustering Algoritmalarını Uygulama ####################

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, SpectralClustering


def apply_clustering_algorithms(X):

    if hasattr(X, "toarray"):
        X = X.toarray()

    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X)
    
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(X)
    
    hierarchical = AgglomerativeClustering(n_clusters=5)
    hierarchical_labels = hierarchical.fit_predict(X)
    
    mean_shift = MeanShift()
    mean_shift_labels = mean_shift.fit_predict(X)

    spectral = SpectralClustering(n_clusters=5, affinity='nearest_neighbors', random_state=42)
    spectral_labels = spectral.fit_predict(X)
    
    return {
        "kmeans": kmeans,
        "dbscan": dbscan,
        "hierarchical": hierarchical_labels,
        "mean_shift": mean_shift_labels,
        "spectral": spectral_labels
    }


#  Kümeleme sonuçlarını DataFrame'e Ekleme

def add_clustering_results(df, clustering_results, X):
    df['KMeans_Cluster'] = clustering_results['kmeans'].predict(X)
    df['KMeans_Cluster'] = df['KMeans_Cluster'].map({i: class_names[i] for i in range(5)})

    df['DBSCAN_Cluster'] = clustering_results['dbscan'].labels_
    df['DBSCAN_Cluster'] = df['DBSCAN_Cluster'].apply(lambda x: 'outlier' if x == -1 else f'cluster {x}')

    df['Hierarchical_Cluster'] = clustering_results['hierarchical']
    df['Hierarchical_Cluster'] = df['Hierarchical_Cluster'].map({i: class_names[i] for i in range(5)})

    df['MeanShift_Cluster'] = clustering_results['mean_shift']
    unique_labels = df['MeanShift_Cluster'].unique()
    label_map = {label: class_names[i % len(class_names)] for i, label in enumerate(unique_labels)}
    df['MeanShift_Cluster'] = df['MeanShift_Cluster'].map(label_map)

    df['Spectral_Cluster'] = clustering_results['spectral']
    df['Spectral_Cluster'] = df['Spectral_Cluster'].map({i: class_names[i] for i in range(5)})

    return df



# LDA ve NMF Modellerini Uygulama
from sklearn.decomposition import LatentDirichletAllocation, NMF

def apply_lda(df, X, n_topics=5):
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_topics = lda.fit_transform(X)
    df['LDA_Topic'] = lda_topics.argmax(axis=1)
    df['LDA_Topic'] = df['LDA_Topic'].map({i: class_names[i] for i in range(n_topics)})
    return df, lda

def apply_nmf(df, X, n_topics=5):
    nmf = NMF(n_components=n_topics, random_state=42)
    nmf_topics = nmf.fit_transform(X)
    df['NMF_Topic'] = nmf_topics.argmax(axis=1)
    df['NMF_Topic'] = df['NMF_Topic'].map({i: class_names[i] for i in range(n_topics)})
    return df, nmf

# Kümeleme ve Konu Modelleme modellerinin Performans Karşılaştırması
def calculate_ari_nmi(df):
    results = []
    methods = [
        'KMeans_Cluster', 'DBSCAN_Cluster', 'Hierarchical_Cluster',
        'MeanShift_Cluster', 'Spectral_Cluster', 'LDA_Topic', 'NMF_Topic'
    ]

    for method in methods:
        try:
            ari = adjusted_rand_score(df['label'], df[method])
            nmi = normalized_mutual_info_score(df['label'], df[method])
            results.append({'Model': method.replace('_Cluster', '').replace('_Topic', ''), 'ARI': ari, 'NMI': nmi})
        except Exception as e:
            print(f"Hata ({method}): {e}")
    
    return pd.DataFrame(results)


# 1. Vektörleştirme sonrası X hazır olmalı
clustering_results = apply_clustering_algorithms(X)

# 2. LDA ve NMF
df, lda_model = apply_lda(df, X)
df, nmf_model = apply_nmf(df, X)

# 3. Clustering Sonuçlarını Ekle 
df = add_clustering_results(df, clustering_results, X)

# 4. Performans Değerlendirme
results_df = calculate_ari_nmi(df)
print(results_df)


# Performans sonuçlarını CSV dosyasına kaydet
results_df.to_csv("clustering_topic_modeling_results.csv", index=False)

print("Sonuçlar 'clustering_topic_modeling_results.csv1' dosyasına kaydedildi.")


























# import pandas as pd

# def predict_input_text_interactive(vectorizer, lda_model, nmf_model, clustering_models):
#     # Kullanıcıdan metin al
#     user_input = input("Lütfen bir metin girin: ")

#     # Vektörleştir
#     X_input_sparse = vectorizer.transform([user_input])
#     X_input_dense = X_input_sparse.toarray()

#     # LDA & NMF
#     lda_topic_distribution = lda_model.transform(X_input_sparse)[0]
#     nmf_topic_distribution = nmf_model.transform(X_input_sparse)[0]

#     lda_topic = int(lda_topic_distribution.argmax())
#     nmf_topic = int(nmf_topic_distribution.argmax())

#     # Kümeleme
#     clustering_results = {}
#     for name, model in clustering_models.items():
#         try:
#             if hasattr(model, "predict"):
#                 label = int(model.predict(X_input_dense)[0])
#             else:
#                 # Agglomerative, MeanShift, Spectral gibi fit_predict kullananlar için yeniden fit gerekebilir.
#                 label = int(model.fit_predict(X_input_dense)[0])
#         except Exception as e:
#             label = f"Hata: {str(e)}"
#         clustering_results[name] = label

#     # DataFrame oluştur
#     result = {
#         "LDA_Topic": lda_topic,
#         "NMF_Topic": nmf_topic,
#         "KMeans_Cluster": clustering_results.get("kmeans"),
#         "DBSCAN_Cluster": clustering_results.get("dbscan"),
#         "Hierarchical_Cluster": clustering_results.get("hierarchical"),
#         "MeanShift_Cluster": clustering_results.get("mean_shift"),
#         "Spectral_Cluster": clustering_results.get("spectral"),
#     }

#     df_result = pd.DataFrame([result])
#     print("\nTahmin Sonuçları:")
#     print(df_result)
#     return df_result


# predict_input_text_interactive(
#     vectorizer=vectorizer,
#     lda_model=lda_model,
#     nmf_model=nmf_model,
#     clustering_models=clustering_results  # Burada clustering_results'ü geçiyoruz
# )

