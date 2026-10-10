#import bölümü
import nltk

from nltk.corpus import stopwords

nltk.download("stopwords")

#ingilizce stopwords kaldırımı

stopwords_eng= set(stopwords.words("english")) #ingilizce stopwords db

text="Hello world, how are you? Can you give me a hug?"

text_list_eng=text.lower().split()
cleaned_text_eng=[word for word in text_list_eng if word not in stopwords_eng]

#%% turkce stopwords kaldırımı

stopwords_tr= set(stopwords.words("turkish")) #set hepsinden benzersiz olanı yazar

metin="Merhaba dünya! Benim çok ama çok uykum var."

metin_list= metin.split()

temiz_metin=[word for word in metin_list if word.lower() not in stopwords_tr]

#%% seçtiğimiz kelimeleri çıkartma

stopwords_selected=["bir","bu", "da", "mi", "kim", "ki"]

metin="Kim güzel bir uyku istemez ki ?"

temiz_metin2= [word for word in metin.split() if word.lower() not in stopwords_selected]
