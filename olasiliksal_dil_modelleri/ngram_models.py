#import libraries
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

from collections import Counter

#veriseti oluştur

corpus=[
        "Ben seni seviyorum",
        "Ben seni özlüyorum",
        "Ben seni istiyorum",
        "Ben sana borçluyum",
        "Ben sana vurgun",
        "Sen beni seviyorsun",
        "Onlar beni seviyorsun ben sana vurgun",
        "Onlar sana vurgun sen beni seviyorsun"]

#tokenize işlemi

tokens= [word_tokenize(sentence.lower()) for sentence in corpus]

#bigram
bigrams= []

for token_list in tokens:
    bigrams.extend(list(ngrams(token_list, 2)))

bigrams_freq= Counter(bigrams)
#trigram
trigrams=[]

for token_list in tokens:
    trigrams.extend(list(ngrams(token_list,3)))

trigrams_freq=Counter(trigrams)
#model test

#ben seni bigramından sonra seviyorum gelme olasılığını hesaplayalım
bigram_test= ("ben", "seni")

#seviyorum olma olasılığı

test_prob_seviyor= trigrams_freq[("ben","seni","seviyorum")]/bigrams_freq[(bigram_test)]

print(f"seviyorum kelimesinin gelme olasılığı: {test_prob_seviyor}")

#ben sana vurgun gelme bigramının gelme olasılığı
bigram_test2=("ben", "sana")

#vurgun olasılığı
test_prob_vurgun= trigrams_freq[("ben","sana","vurgun")]/bigrams_freq[bigram_test2]

print(f"vurgun kelimesinin gelme olasılığı: {test_prob_vurgun}")