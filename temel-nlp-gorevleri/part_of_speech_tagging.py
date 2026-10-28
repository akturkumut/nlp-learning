import spacy

nlp = spacy.load("en_core_web_sm")

sentence = "How do you feel now"

doc = nlp(sentence)

for token in doc:
    print(token.text, token.pos_)
    
"""
How SCONJ
do AUX
you PRON
feel VERB
now ADV
"""