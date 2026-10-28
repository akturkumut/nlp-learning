# -*- coding: utf-8 -*-
# https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv
"""
Metin Sınıflandırma - Text classsification

Spam csv -> Spam and ham -> binary classification with decision tree
"""

#import lib

import pandas as pd

#veri setini yükle

data = pd.read_csv("spam_nlp.csv", encoding= "latin-1")
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1, inplace= True)
data.columns = ["label", "text"]


#EDA: Keşifsel veri analizi
print(data.isna().sum())

#%% text cleaning and preprocessing: ozel karakter, lowercase, tokenization, stopwords, lemmatization

import nltk

nltk.download("stopwords") # cok kullanilan v eanlam tasimayan sozcukleri metin icerisinden cikartalim
nltk.download("wordnet") # lemma bulmak icin gerekli olan veriseti
nltk.download("omw-1.4") # wordnete ait farkli dillerin kelime anlamlarini iceren bir veri seti

import re 
from nltk.corpus import stopwords  # stopwords lerden kurtulmak icin
from nltk.stem import WordNetLemmatizer # lemmatization

text = list(data.text)
lemmatizator = WordNetLemmatizer()

corpus = []

for i in range(len(text)):
    
    r = re.sub("[^a-zA-Z]"," ",text[i]) #ozel karakterleri çıkar
    
    r = r.lower() #küçük harf
    
    r = r.split() #tokenize
    
    r = [word for word in r if word not in stopwords.words("english")] #stopwordlerden kurtul
    
    r = [lemmatizator.lemmatize(word)for word in r]
    
    r = " ".join(r)
    
    corpus.append(r)

data["text2"] = corpus


#%% model training and evaluation

X = data["text2"]
y = data["label"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state=42 )

# feature extraction: bag of words

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)

# classifier training: model training and evaluation
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

dt.fit(X_train_cv,y_train) # egitim

X_test_cv = cv.transform(X_test)

# prediction

prediction = dt.predict(X_test_cv)

from sklearn.metrics import confusion_matrix

c_matrix = confusion_matrix(y_test, prediction)

accuracy = 100*((sum(sum(c_matrix)) - c_matrix[1,0] - c_matrix[0,1]) /sum(sum(c_matrix)))

print(f"Basarim: {accuracy}")





















