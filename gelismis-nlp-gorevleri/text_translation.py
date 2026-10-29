from transformers import MarianMTModel, MarianTokenizer

model_name ="Helsinki-NLP/opus-mt-en-de"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = "Hello, where are you from ?"

translated_text = model.generate(**tokenizer(text, return_tensors = "pt", padding= True))

translated_text = tokenizer.decode(translated_text[0], skip_special_tokens= True)

print(f"Translated text: {translated_text}")