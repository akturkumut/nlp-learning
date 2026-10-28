#import lib
import nltk
from nltk.wsd import lesk

nltk.download("wordnet")
nltk.download("own-1.4")
nltk.download("punkt")

sentence1 = "I was in central bank in London for deposit money"
word1 = "bank"

sense1 = lesk(nltk.word_tokenize(sentence1),word1)

print(f" Cumle1: {sentence1}")

print(f" Kelime1: {word1}")

print(f" Anlam: {sense1.definition()}")

"""
 Cumle1: I was in central bank in London for deposit money
 Kelime1: bank
 Anlam: a container (usually with a slot in the top) for keeping money at home
"""

sentence2 = "I tried to rob the bank"
word2 = "bank"

sense2 = lesk(nltk.word_tokenize(sentence2), word2)

print(f" Cumle2: {sentence2}")

print(f" Kelime2: {word2}")

print(f" Anlam: {sense2.definition()}")

"""
 Cumle2: I tried to rob the bank
 Kelime2: bank
 Anlam: a slope in the turn of a road or track; the outside is higher than the inside in order to reduce the effects of centrifugal force
"""