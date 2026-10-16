#importlar
import pandas as pd
import matplotlib.pyplot as plt
import re

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

import nltk

from nltk.corpus import stopwords

nltk.download("stopwords")

#ingilizce stopwords kaldırımı

stopwords_eng= set(stopwords.words("english")) #ingilizce stopwords db
#veriseti yükleme
df= pd.read_csv("IMDB Dataset.csv")
documents = df["review"]

#metin temizleme
def clean_text(text):
    text = text.lower() #kucuk harfe cevir
    
    text = re.sub(r"\d+","", text) # sayıları temizle
    
    text = re.sub(r"[^\w\s]","", text) #ozel karakterleri temizle
    
    text =" ".join([word for word in text.split() if len(word) > 2 and word not in stopwords_eng])
    
    return text

cleaned_docs= [clean_text(doc) for doc in documents]

#tokenization

tokenized_docs = [simple_preprocess(doc) for doc in cleaned_docs]

#%%

#word2vec modeli
model = Word2Vec(sentences=tokenized_docs, vector_size=50, window=5, min_count=2, sg=0)
word_vectors = model.wv

words= list(word_vectors.index_to_key[:200])

vectors = [word_vectors[word] for word in words]

#clustering
kmeans = KMeans(n_clusters=2)
kmeans.fit(vectors)

clusters=kmeans.labels_

#pca 50 -> 2
# PCA 50 -> 3 (3 Boyutlu)
pca = PCA(n_components=3)
reduced_vectors = pca.fit_transform(vectors)

# 3 Boyutlu Görselleştirme için Ekseni Tanımlayalım
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Kelime vektörlerini çiz
ax.scatter(
    reduced_vectors[:, 0],
    reduced_vectors[:, 1],
    reduced_vectors[:, 2],
    c=clusters,
    cmap="viridis",
    alpha=0.5,
)

# K-Means Merkezlerini Dönüştür ve Çiz (size yerine s=150 kullanıldı)
centers = pca.transform(kmeans.cluster_centers_)
ax.scatter(
    centers[:, 0],
    centers[:, 1],
    centers[:, 2],
    s=150,
    c="red",
    marker="x",
    label="Center",
)
ax.legend()

# Figür üzerine kelimelerin yazılması (Çok fazla kelime olduğu için ilk 200 tanesini yazdıralım)
for i, word in enumerate(words[:200]):
  ax.text(
      reduced_vectors[i, 0],
      reduced_vectors[i, 1],
      reduced_vectors[i, 2],
      word,
      fontsize=6,
  )

ax.set_title("Word2Vec & KMeans (IMDB)")
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_zlabel("Component 3")
plt.show()





























