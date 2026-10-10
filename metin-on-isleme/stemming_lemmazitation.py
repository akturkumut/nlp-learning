import nltk

nltk.download("wordnet")

from nltk import PorterStemmer

words=["run", "runner", "ran", "runs", "better", "go", "went"]
print(f"base words: {words}")

stemmer = PorterStemmer()

stems = [stemmer.stem(w) for w in words] """
Bu kullanımın adı List Comprehension (Liste Kavramı / Liste Üreteci)'dir.

Python'da var olan bir veri dizisi (liste, demet vb.)
üzerinde dönerek tek satırda yeni bir liste oluşturmaya yarayan kısa ve pratik
bir yazım şeklidir.
"""

print(f"Stems: {stems}")
#%% lemmazitation işlemi

from nltk.stem import WordNetLemmatizer

words=["run", "runner", "ran", "runs", "better", "go", "went"]

lemmatizator= WordNetLemmatizer()

lemmas= [lemmatizator.lemmatize(w, pos="v") for w in words] 

print(f"lemmas: {lemmas}")