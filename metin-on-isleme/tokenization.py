import nltk
#yeni sürümde artık tokenizasyon işlemi için punkt_tab kütüphanesi kullanılıyor.
nltk.download("punkt_tab")

text="Hello world! How are you? Hello, hi..."
#kelime tokenizasyonu
words_tokens= nltk.word_tokenize(text)
#cümle tokenizasyonu
sentence_tokens= nltk.sent_tokenize(text)