#import libraries
from nltk.classify import MaxentClassifier

#verisetini tanımla
train_data = [
    ({"love":True , "happy": True, "amazing": True, "hate": False}, "positive"),
    ({"sad": True, "terrible": True, "bad": True , "like":False}, "negative"),
    ({"perfect": True, "great": True, "hate": False}, "positive"),
    ({"sad": True, "depressed": True, "love": False},"negative"),
    ({"horrible": True, "sad": True, "love": True}, "tragic")
    ]
#train max ent classifier
classifier= MaxentClassifier.train(train_data, max_iter=10)
#yeni cümle ile test 
test_sentence = "This movie is sad and horrible but I love it"

features= {word: (word in test_sentence.lower().split()) for word in ["love", "happy","amazing", "hate", "sad","terrible"
                                                                      ,"bad","like","perfect","great","depressed","horrible"]}

label= classifier.classify(features)

print(label)