#import libs
from transformers import BertTokenizer, BertModel
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

#tokenizer and model
model_name = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

#veri üretimi
docs = ["Data science is a part of computer engineering",
        "English is the most common language in the world",
        "Chinese is the second most common language",
        "Turks have great historical cannons",
        "Space ships are expensive"
        ]

query = "What is data science"

#bert ile bilgi getirimi

def get_embedding(text):
    inputs = tokenizer(text, return_tensors = "pt", truncation = True, padding = True)
    
    #modeli çalıştır
    outputs = model(**inputs)
    
    #son gizli katmanı al
    last_hidden_state = outputs.last_hidden_state
    
    #metin temsili oluştur
    embedding = last_hidden_state.mean(dim=1)
    
    #vektorü numpy olarak return et
    return embedding.detach().numpy()


doc_embeddings = np.vstack([get_embedding(text) for text in docs])

query_embedding = get_embedding(query)

#cosine benzerlik ile belgeler arasındaki benzerliği hesaplayalım
similarities = cosine_similarity(query_embedding, doc_embeddings)

#her belgenin benzerlik skoru
for i, score in enumerate(similarities[0]):
    print(f"Document: {docs[i]} \n Similarity score: {score}\n")



# query = "What is data science"

"""
Document: Data science is a part of computer engineering 
 Similarity score: 0.6864748001098633

Document: English is the most common language in the world 
 Similarity score: 0.5344986915588379

Document: Chinese is the second most common language 
 Similarity score: 0.45072537660598755

Document: Turks have great historical cannons 
 Similarity score: 0.49257928133010864

Document: Space ships are expensive 
 Similarity score: 0.6620578765869141
"""

most_similar_index = similarities.argmax()

print(f"Most similar document: {docs[most_similar_index]}")

"""
Most similar document: Data science is a part of computer engineering
"""

















