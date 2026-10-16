# Kütüphanelerin içe aktarılması
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

stopwords_eng= set(stopwords.words("english")) #ingilizce stopwords db

# Veri setinin içeriye aktarılması (IMDB film yorumları veri setini yüklüyoruz)
df = pd.read_csv("IMDB Dataset.csv")

# Metin verilerini ve etiketleri (sentiment: pozitif/negatif) değişkenlere ayırma
documents = df["review"]
labels = df["sentiment"]  # pozitif negatif labellar

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

# Veri setindeki tüm yorumlar üzerine yukarıdaki temizleme fonksiyonunu uyguluyoruz
cleaned_doc = [clean_text(row) for row in documents]

#%% Bag-of-Words (BoW) İşlemleri

# Vectorizer tanımlama: Metinleri kelime frekanslarına göre sayısallaştıracak 
# CountVectorizer sınıfından bir nesne (araç) oluşturuyoruz.
vectorizer = CountVectorizer()

# fit_transform(): İki adımı arka arkaya çalıştırır.
# - fit aşaması: Metinlerdeki tüm benzersiz kelimeleri tarar ve bir kelime sözlüğü oluşturur.
# - transform aşaması: Cümlelerde bu kelimelerin kaçar kez geçtiğini hesaplar ve sayısal matrise döker.
# Not: Bellek tasarrufu için veri setinin sadece ilk 75 yorumunu seçip X değişkenine atıyoruz.
X = vectorizer.fit_transform(cleaned_doc[:75])

# get_feature_names_out(): Vectorizer'ın hafızasında tuttuğu o kelime sözlüğünü 
# alfabetik sıraya dizerek bir NumPy dizisi olarak dışarı aktarır.
feature_names = vectorizer.get_feature_names_out()

# toarray(): fit_transform sonucunda bellekten tasarruf etmek için optimize edilmiş 
# sıkıştırılmış (sparse) matrisi, insan gözüyle incelenebilir klasik bir iki boyutlu diziye çevirir.
vektor_temsili = X.toarray()

# pd.DataFrame(...): Sayısal matrisi satır ve sütunlarıyla birlikte gösteren 
# Pandas tablo formatına dönüştürür (Sütun isimleri kelime sözlüğümüz olur).
df_bow = pd.DataFrame(vektor_temsili, columns=feature_names)

# X.sum(axis=0): Matristeki sütunlar boyunca (yukarıdan aşağıya) toplama yapar. 
# .A1 metodu ise bu sonucu tek boyutlu düz bir diziye çevirir. Böylece her kelimenin 
# toplam kaç kez geçtiğini buluruz. zip() ile kelimeler ve frekansları eşleştirilip sözlüğe aktarılır.
word_counts = X.sum(axis=0).A1
word_freq = dict(zip(feature_names, word_counts))

# Counter().most_common(5): Sözlükteki kelime frekanslarını analiz eder ve 
# en çok tekrar eden (en yüksek sayıya sahip) ilk 5 kelimeyi büyükten küçüğe sıralayarak seçer.
most_common_5_words = Counter(word_freq).most_common(5)
print(f"Most common 5 words are : {most_common_5_words}")