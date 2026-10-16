#WORD embedding modelleri
#kütüphaneleri import etmek
#import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA #principle component analysis, dimension reduction

from gensim.models import Word2Vec, FastText
from gensim.utils import simple_preprocess
#verisetini oluşturmak
sentences= [
    "Bilgisayar mühendislerinin iş bulması çok zor.",
    "İş bulmak çok zor.",
    "İş arıyorum.",
    "Işsizlik insanı depresyona sokar.",
    "Nlp öğrenmek yardımcı olabilir."]

tokenized_sentences= [simple_preprocess(sentence)for sentence in sentences]

##word2vec
model_word2vec= Word2Vec(sentences= tokenized_sentences, vector_size=50, window=5, min_count=1,sg=0)
#fasttext
model_fasttext= FastText(sentences= tokenized_sentences, vector_size=50, window=5, min_count=1,sg=0)

#görselleştirme

def plot_word_embed(model, title):
    word_vectors= model.wv
    
    words= list(word_vectors.index_to_key[:1000])
    vectors = [word_vectors[word] for word in words]
    
    #PCA 
    pca= PCA(n_components=3)
    reduced_vectors= pca.fit_transform(vectors)
    
    #3d görselleştirme
    fig= plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection ="3d")
    
    #vektorleri çiz
    ax.scatter(reduced_vectors[:,0], reduced_vectors[:,1], reduced_vectors[:,2])
    
    for i, word in enumerate(words):
        ax.text(reduced_vectors[i,0], reduced_vectors[i,1], reduced_vectors[i,2], word, fontsize =12)
    
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    plt.show()
    
plot_word_embed(model_word2vec, "Word2Vec")
plot_word_embed(model_fasttext, "FastText")

        
        
        
        
        
        
        
        
        