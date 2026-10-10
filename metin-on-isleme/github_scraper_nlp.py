from bs4 import BeautifulSoup
import requests
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Gerekli NLTK paketlerini indirelim
nltk.download("stopwords")
nltk.download("punkt_tab")
nltk.download("wordnet")

url = 'https://github.com/akturkumut/Automated-Exam-Scoring-LLM'
response = requests.get(url)

# HTML'yi BeautifulSoup ile parse edilebilir hale getiriyoruz
soup = BeautifulSoup(response.text, 'html.parser')

# GitHub'da proje açıklamaları veya README genelde 'article etiketi' ve 'markdown-body' sınıfı içinde yer alır
readme_alani = soup.find('article', class_='markdown-body')

if readme_alani:
    # Etiketlerden arındırılmış saf metni alıyoruz
    temiz_metin = readme_alani.get_text()
    print("\n--- README İçeriği ---")
    print(temiz_metin[:500]) # Çok uzun olmasın diye ilk 500 karakteri yazdıralım
else:
    print("İstenen alan bu sayfada bulunamadı.")
    temiz_metin = ""

if temiz_metin:
    # Noktalama işaretlerini kaldıralım
    temiz_metin = temiz_metin.translate(str.maketrans("", "", string.punctuation))

    # Boşlukları temizleyelim, küçük harfe çevirelim
    temiz_kelimeler = temiz_metin.lower().split()

    # Stopwords'leri kaldıralım (GitHub reposu İngilizce olduğu için 'english' seçiyoruz)
    ingilizce_stopwords = stopwords.words("english")
    temiz_kelimeler = [word for word in temiz_kelimeler if word not in ingilizce_stopwords]

    temiz_metin = " ".join(temiz_kelimeler)

    # Özel karakterleri kaldıralım
    temiz_metin = re.sub(r"[^A-Za-z0-9\s]", "", temiz_metin)

    # Tokenizasyon işlemi
    metin_tokens = nltk.word_tokenize(temiz_metin)

    # Lemmatization işlemi
    lemmatizator = WordNetLemmatizer()
    lemmas = [lemmatizator.lemmatize(w, pos="v") for w in metin_tokens] 

    print("\n--- İşlenmiş İlk 10 Lemma ---")
    print(lemmas[:10])