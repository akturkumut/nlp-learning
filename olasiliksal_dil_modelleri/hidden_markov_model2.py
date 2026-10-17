#import libraries
import nltk
from nltk.tag import hmm
from nltk.corpus import conll2000

#import datasets
nltk.download("conll2000")

train_data = conll2000.tagged_sents("train.txt")
test_data = conll2000.tagged_sents("test.txt")

#hmm train
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

#test
test_sentence = "Master is a teacher".split()

tags = hmm_tagger.tag(test_sentence)
print(tags)
"""
[('Master', 'NN'), ('is', 'NN'), ('a', 'NN'), ('teacher', 'NN')]
"""
#%% Laplace (Add-1) düzeltmesi

import nltk
from nltk.tag import hmm
from nltk.corpus import conll2000
from nltk.probability import LaplaceProbDist  # Laplace düzeltmesi için

nltk.download("conll2000")

train_data = conll2000.tagged_sents("train.txt")

# HMM Eğiticisine Laplace Smoothing Estimator'ını tanımlıyoruz
# Bu sayede görülmeyen geçişlere ve kelimelere 0 yerine çok küçük paylar verilir.
trainer = hmm.HiddenMarkovModelTrainer()

# NLTK'de estimator parametresi ile Laplace dağılımını entegre edebiliriz:
hmm_tagger = trainer.train(
    train_data, 
    estimator=lambda fd, bins: LaplaceProbDist(fd, bins)
)

# Test cümlemiz
test_sentence = "Master is a teacher".split()

tags = hmm_tagger.tag(test_sentence)
print(tags)
"""
[('Master', 'PRP'), ('is', 'VBZ'), ('a', 'DT'), ('teacher', 'NN')]
"""

#%%faklı bir örnek nltk 
#NLTK'nin varsayılan pos_tag fonksiyonu Perceptron tabanlı gelişmiş bir model kullanır.
import nltk
from nltk.tokenize import word_tokenize

# NLTK'nin hazır modelini indirmek gerekebilir (eğer indirilmediyse):
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')


# Test cümlemiz
test_sentence = word_tokenize("Master is a teacher")

# Hazır fonksiyonu çağırıyoruz
tags = nltk.pos_tag(test_sentence)

print(tags)
"""
[('Master', 'NN'), ('is', 'VBZ'), ('a', 'DT'), ('teacher', 'NN')]
"""