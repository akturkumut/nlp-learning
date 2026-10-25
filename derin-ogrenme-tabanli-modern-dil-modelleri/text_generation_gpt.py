# -*- coding: utf-8 -*-
"""
Text generation with GPT-2
"""

#import lib
from transformers import GPT2LMHeadModel, GPT2Tokenizer

#model name
model_name = "gpt2"

#tokenizer and model tanımlanması
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

#devamı üretilecek metnin tanımlanması

text = "Istanbul is a great"

#tokenization

inputs = tokenizer.encode(text, return_tensors="pt")

#text generation

outputs = model.generate(inputs, max_length = 55)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#uretilen metni print ettirelim
print(generated_text)

"""
Istanbul is a great place to live, but it's also a place where you can't go to a party without being in a bad mood.

"I'm not sure if I'll be able to go to a party without being in a bad mood. I
"""
