#import kütüphaneler
import pandas as pd
#import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

#belge oluştur
doc = [
       "Bu bilgisayar çok yavaş",
       "Bu bilgisayar çok iyi",
       "Bu elma çok tatlı"]

#vektorizer tanımla
tfidf_vectorizer= TfidfVectorizer()

#metinleri sayısal hale çevir
X = tfidf_vectorizer.fit_transform(doc)
#kelime kümesini incele
feature_names = tfidf_vectorizer.get_feature_names_out()
#vektör temsilini incele
vektor_temsili= X.toarray()

df= pd.DataFrame(vektor_temsili, columns=feature_names)
#ortalama tf-idf değerlerini hesapla
tf_idf= df.mean(axis=0)