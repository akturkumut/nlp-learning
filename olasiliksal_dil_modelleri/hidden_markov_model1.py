"""
Part of Speech POS: kelimelerin uygun sözcük türünü bulma çalısması

I(Zamir) am a teacher(isim)
"""

#import libraries
import nltk
from nltk.tag import hmm
#ornek training data tanımla
train_data = [[("I","PRP"), ("am", "VBP"), ("a","DT"), ("student", "NN")], [("You","PRP"), ("are", "VBP")
              ,("a", "DT"), ("teacher", "NN")], [("He", "PRP"), ("is", "VBP"), ("a","DT"),("master", "NN")]]
# model training

trainer= hmm.HiddenMarkovModelTrainer()
hmm_tagger= trainer.train(train_data)

#yeni bir cümle tanımla ve cumlenin içerisindeki her bir sözcüğü etiketle
test_sentence="He is a teacher".split()

tags = hmm_tagger.tag(test_sentence)
print(f"yeni cümle tagler: {tags}")

#yeni başka cümle- yanlış output

test_sentence2= "master is a teacher".split()

tags2 = hmm_tagger.tag(test_sentence)
print(f"yeni cümle tagler: {tags2}")


