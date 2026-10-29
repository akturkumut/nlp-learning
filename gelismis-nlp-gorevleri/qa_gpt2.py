#import lib
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

import warnings
warnings.filterwarnings("ignore")

model_name= "gpt2"

model= GPT2LMHeadModel.from_pretrained(model_name)

tokenizer= GPT2Tokenizer.from_pretrained(model_name)

def generate_answer(context, question):
    input_text = (
            f"Read the text below and answer the question using ONLY the provided text.\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
    
    inputs = tokenizer.encode(input_text, return_tensors = "pt")
    
    #modeli çalıştır
    with torch.no_grad():
        outputs = model.generate(inputs, max_length = 380)
        
    #uretilen metni decode edelim
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens= True)
    
    #yanitlari ayiklama
    
    # tanitlari ayiklayalim
    answer = answer.split("Answer:")[-1].strip()
    
    return answer

question = "What color was the crystal that Dr. Aris found in the cave?"

context = """
On a rainy Tuesday evening in the remote mountains of Zephyria, Dr. Aris embarked on an archaeological expedition. 
Inside the damp walls of the forgotten Obsidian Cave, he discovered a hidden chamber. 
In the center of the room sat a glowing obsidian pedestal holding a pulsating, luminous purple crystal. 
According to the ancient carvings on the wall, this artifact was known as the Heart of Zephyria. 
Dr. Aris carefully packed the purple crystal into his leather satchel and exited the cave before midnight.
"""

answer = generate_answer(context, question)
print(f"Answer: {answer}")


