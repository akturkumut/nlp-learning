from sklearn.feature_extraction.text import CountVectorizer

# 1. Adım: Modelimize girdi olarak vereceğimiz ham metinlerden (belgelerden) 
# oluşan örnek bir veri seti hazırlıyoruz.
documents = [
    "telefon cebimde",
    "telefon şarjda"
]

# 2. Adım: Metinlerdeki kelimeleri sayarak sayısal matrise dönüştürecek 
# olan CountVectorizer nesnesini (aracını) tanımlıyoruz.
vectorizer = CountVectorizer()

# 3. Adım: fit_transform() fonksiyonu iki işlevi aynı anda yapar:
# - fit: Metinlerdeki tüm benzersiz kelimeleri öğrenir ve bir sözlük oluşturur.
# - transform: Cümlelerdeki kelime frekanslarını hesaplayarak sayısal bir matrise çevirir.
# Elde ettiğimiz bu sayısal veri matrisini ileride modellere vermek üzere X değişkenine atıyoruz.
X = vectorizer.fit_transform(documents)

# 4. Adım: Vectorizer nesnesinin hafızasında öğrendiği benzersiz kelime 
# kümesini (sözlüğü) alfabetik sırayla dışarı aktarıyoruz.
kelime_kumesi = vectorizer.get_feature_names_out()

# 5. Adım: fit_transform sonucunda bellekten tasarruf etmek için sıkıştırılmış formatta 
# gelen matrisi, insan gözüyle rahatça okunabilen standart bir diziye (array) çeviriyoruz.
vector_temsili = X.toarray()