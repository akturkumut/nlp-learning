#import libraries
from transformers import AutoModel, AutoTokenizer
import torch

#modeli ve tokenizer'ı yükle
model_name ="bert-base-uncased"

tokenizer_bert = AutoTokenizer.from_pretrained(model_name)
model_bert = AutoModel.from_pretrained(model_name)

#input verisini oluştur
text= "Bert is a transformers model."

#metni tokenlara çevir
inputs = tokenizer_bert(text, return_tensors="pt") #çıktı pytorch tensor olarak return edilir

#modeli kullanarak metin temsili oluştur

with torch.no_grad():
    outputs = model_bert(**inputs)
    
#model çıktısından son gizli durumu al

last_hidden_state = outputs.last_hidden_state #tüm token çıktılarını almak için

#ilk tokenin embeddingini al ve ekrana bas

first_token_embedding = last_hidden_state[0,0,:].numpy()

print(f"metin temsili {first_token_embedding}")