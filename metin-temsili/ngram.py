#import libraries
from sklearn.feature_extraction.text import CountVectorizer

#verisetini oluştur
doc= [
      "Bu bir ngram çalışmasıdır.",
      "Bu çalışma spyder uygulamasında yapılmaktadır."]

#vektorizer tanımla
#unigram
vektorizer_unigram= CountVectorizer(ngram_range=(1,1))

X_unigram = vektorizer_unigram.fit_transform(doc)
feature_name_unigram= vektorizer_unigram.get_feature_names_out()
#bigram
vektorizer_bigram= CountVectorizer(ngram_range=(2,2))
X_bigram= vektorizer_bigram.fit_transform(doc)
feature_names_bigram= vektorizer_bigram.get_feature_names_out()

#trigram
vektorizer_trigram= CountVectorizer(ngram_range=(3,3))
X_trigram= vektorizer_trigram.fit_transform(doc)
feature_names_trigram= vektorizer_trigram.get_feature_names_out()
#sonucları incele5
print(f"Unigram sonuçları: {X_unigram.toarray()}")
print(f"Bigram sonuçları: {X_bigram.toarray()}")
print(f"Trigram sonuçları: {X_trigram.toarray()}")

