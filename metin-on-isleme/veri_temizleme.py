 # boşlukları kaldırma
text = "Hello,         World!     2026"

# 1. Adım: text.split() çalışır
kelimeler = text.split()
print(kelimeler)
# Çıktı: ['Hello,', 'World!', '2026']

# 2. Adım: " ".join(...) listeyi aralara tek boşlık koyarak birleştirir
cleaned_text = " ".join(kelimeler)
print(cleaned_text)
# Çıktı: Hello, World! 2026

#%% büyük küçük harf değişimi

text="Hello World"

cleaned_text2= text.lower()
#%% noktalama işaretlerini kaldır

import string

text="Hello, World! 2035"

cleaned_text3=text.translate(str.maketrans("","",string.punctuation))
#%% Ozel karakterleri kaldır
import re

text="Hello, World!@#& 2035"

cleaned_text4= re.sub(r"[^A-Za-z0-9\s]", "", text)
#%% Yazim hatalarini düzelt
from textblob import TextBlob

text="Hellio Wirld! 2035"

cleaned_text5= TextBlob(text).correct()
print(f"Cleaned text: {cleaned_text5}")
#%% html ya da url etiketlerini kaldır
from bs4 import BeautifulSoup

text="<div>Hello World! 2035</div>"

cleaned_text6=BeautifulSoup(text,"html.parser").get_text()
print(f"Cleaned text: {cleaned_text6}")