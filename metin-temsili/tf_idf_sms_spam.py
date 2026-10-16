#import lib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords

#veri setini indir
df= pd.read_csv("sms_spam.csv")
#veri temizle
stopwords_eng= set(stopwords.words("english"))

# Veri setinin temizlenmesi - Metin temizleme fonksiyonu
def clean_text(text):
    
    text = text.lower()
    
    # re.sub(r"\d+", "", text): Düzenli ifade (regex) kullanarak metindeki tüm rakamları siler.
    # Çalışma mantığı: "\d+" kalıbı 0-9 arasındaki sayıları arar, bulduğu sayıları 
    # boş bir karakter ("") ile değiştirerek metinden tamamen çıkarır.
    text = re.sub(r"\d+", "", text)
    
    # re.sub(r"[^\w\s]", "", text): Özel karakterleri ve noktalama işaretlerini temizler.
    # Çalışma mantığı: "[^\w\s]" kalıbı, harf, rakam ve boşluk dışındaki tüm karakterleri 
    # (nokta, virgül, ünlem vb.) bulur ve yok eder.
    text = re.sub(r"[^\w\s]", "", text)
    
    # Önce metni kelimelerine ayıralım
    words = text.split()
    
    # Hem stopword'leri hem de 2 karakterden kısa kelimeleri tek hamlede eleyelim
    filtered_words = [word for word in words if word not in stopwords_eng and len(word) > 2]
    
    # Kalan kelimeleri tekrar tek bir cümle (string) haline getirelim
    text = " ".join(filtered_words)
    
    return text

df.text = [clean_text(row) for row in df.text]

#vektorizer tanımla
vektorizer= TfidfVectorizer()

X = vektorizer.fit_transform(df.text)

#kelime kumesi
feature_names = vektorizer.get_feature_names_out()
#tfidf score hesaplama
tfidf_score= X.mean(axis=0).A1
#df tfidf
df_tfidf= pd.DataFrame({"word":feature_names,"tfidf_score" : tfidf_score })
#sorted
df_tfidf_sorted= df_tfidf.sort_values(by="tfidf_score", ascending=False)
print(df_tfidf_sorted.head(10))