# -*- coding: utf-8 -*-
"""
Spacy kullanarak varlık ismi tanıma
"""

#import lib
import pandas as pd #data frame'e çevirmek için
import spacy

#spacy kullanarak varlık ismi tanıma
nlp = spacy.load("en_core_web_sm") #spacy kütüphanesi ingilizce dil modeli

content = "Today I was working at Berlin Grand Office and I forgot my laptop in room 102."

doc = nlp(content)

for ent in doc.ents:
    #ent.text: etiketlenen kelime
    #ent.label_: kelimenin hangi label ile etiketlendiği
    print(ent.text, ent.label_)
    
"""
Today DATE
Berlin Grand Office ORG
102 CARDINAL
"""
entities = [(ent.text, ent.label_, ent.lemma_) for ent in doc.ents]

#varlik listesini pandas pd ye çevir
df = pd.DataFrame(entities, columns=["text", "label", "lemma"])