"""
Solving a classificaiton problem (Sentiment Analysis) with RNN (Deep Learning based Language Model)

Duygu analizi, pozitif ve negatif etiketli yorumların değerlendirilmesi 

"""

#import libraries
import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#create dataset
data = {
    "text": [
        "Yemekler harikaydı, her şey taze ve lezzetliydi.",
        "Garson çok ilgisizdi, siparişimi unuttular.",
        "Tatlılar gerçekten çok güzeldi, bayıldım!",
        "Yemekler soğuktu ve tadı hiç hoş değildi.",
        "Atmosfer oldukça keyifliydi, tekrar geleceğim.",
        "Fiyatlar biraz yüksekti ama yemekler güzeldi.",
        "Servis kalitesi çok iyiydi, teşekkürler.",
        "Yemek çok geç geldi, sabrım kalmadı.",
        "Lezzetli bir akşam yemeği deneyimledik.",
        "Bu restoranı asla tavsiye etmem, kötüydü.",
        "Mekan çok hoştu, özellikle dekorasyonu.",
        "Yemekler beklediğimden çok daha kötüydü.",
        "Güzel bir akşam geçirdik, teşekkürler.",
        "Yemekler fazlasıyla tuzlu geldi, hiç beğenmedim.",
        "Kahvaltı muhteşemdi, her şeyi denemek istedim.",
        "Fiyatlar oldukça makuldü, çok memnun kaldım.",
        "Garsonlar çok yardımseverdi, teşekkürler.",
        "Yemekler güzel ama servis biraz yavaştı.",
        "Çocuklar için harika bir yer, çok eğlendiler.",
        "Bir daha asla gitmeyeceğim, kötü bir deneyim yaşadım.",
        "Mekanın atmosferi çok keyifliydi.",
        "Yemeklerin tadı harikaydı, özellikle deniz ürünleri.",
        "Şarap menüsü oldukça zengindi, beğendim.",
        "Yemekler sıcak servis edilmedi, hayal kırıklığıydı.",
        "Burgerleri gerçekten çok lezzetliydi.",
        "Tatlıların fiyatı biraz yüksekti ama lezzetliydi.",
        "Hizmet çok yavaştı ama yemekler fena değildi.",
        "Gerçekten güzel bir akşam yemeği deneyimi yaşadık.",
        "Sushi taze ve lezzetliydi, kesinlikle tavsiye ederim.",
        "Garsonlar çok nazik ve yardımseverdi.",
        "Hizmetin daha iyi olmasını beklerdim.",
        "Kahvaltı menüsü oldukça zengindi, çok beğendim.",
        "Yemekler çok lezzetliydi ama servis biraz yavaştı.",
        "Fiyatlar oldukça makuldü, bu kadar iyi hizmete.",
        "Mekan çok temizdi, bu benim için önemli.",
        "Tatlıların çok şekerli olduğunu düşündüm.",
        "Hizmet yavaş ama mekan güzeldi.",
        "Yemeklerin lezzeti harikaydı ama porsiyonlar küçük.",
        "Kendimi çok özel hissettim, teşekkürler.",
        "Güzel bir akşam yemeği, tekrar geleceğim.",
        "Çalışanlar çok güler yüzlüydü.",
        "Pasta çok güzeldi, özellikle çikolatalı.",
        "Biraz beklemek zorunda kaldık ama değdi.",
        "Sadece fiyatlar biraz yüksekti ama lezzet buna değer.",
        "Mekan oldukça kalabalıktı ama hizmet güzel.",
        "Garsonlar çok nazik ama biraz daha hızlı olabilirdi.",
        "Yemeklerin sunumu gerçekten etkileyiciydi.",
        "Yemekler çok lezzetliydi ama garsonlar nazik değildi.",
        "Çok güzel bir akşam yemeği deneyimi yaşadım.",
        "Pasta siparişi verdim ama çok uzun sürdü.",
        "Pizza hamuru çok çıtır ve malzemeler tazeydi.",
        "Masa çok kirliydi, temizlemelerini birkaç kez söylemek zorunda kaldık.",
        "Çalışanların ilgisi ve güler yüzü akşamımızı güzelleştirdi.",
        "Fiyatlar porsiyonlara göre inanılmaz pahalıydı, değmez.",
        "Müzik sesi çok yüksekti, sohbet etmekte zorlandık.",
        "Aparatifler ve ana yemekler tam kıvamında pişmişti, harikaydı.",
        "Sipariş ettiğimiz içecekler ılık geldi, hiç memnun kalmadık.",
        "Rezervasyonumuz olmasına rağmen yarım saat masanın boşalmasını bekledik.",
        "Çorba çok lezzetliydi, baharat dengesi harika ayarlanmıştı.",
        "Manzara eşliğinde harika bir akşam yemeği yedik, herkese öneririm.",
        "Etler lokum gibiydi, pişme derecesi tam istediğim gibiydi.",
        "Garsonların tavrı oldukça kaba ve aceleciydi.",
        "Park yeri bulmak zor oldu ama mekanın hizmeti bu eksiliği unutturdu.",
        "Yemekten sonra ikram edilen çay taze değildi ve bayattı.",
        "Sakin ve huzurlu bir ortamı var, ailece gelmek için çok uygun.",
        "Günün çorbası çok tuzluydu, içmekte güçlük çektik.",
        "Tatlı sunumları görsel olarak harikaydı, tadı da bir o kadar güzeldi.",
        "Masalar birbirine çok yakındı, rahat hareket edemedik.",
        "Hesapta yanlışlık yapılmıştı, düzeltmelerini istemek zorunda kaldık.",
        "Geniş menü seçeneği ve lezzetli alternatifleriyle favori mekanım oldu.",
        "Lahmacunlar fırından çıktığı gibi sıcacık ve çıtır çıtır geldi, bayıldım.",
        "Salatanın içinden maalesef yabancı bir cisim çıktı, çok midem bulandı.",
        "Hızlı servis ve personelin samimi yaklaşımı için teşekkür ederim.",
        "Söylediğimiz kahve tam bir hayal kırıklığıydı, adeta suya benziyordu.",
        "Ortamdaki aydınlatma ve dekorasyon insanı anında rahatlatıyor.",
        "İstediğimiz ana yemek kalmamıştı, sunulan alternatifler de pek tatmin edici değildi.",
        "Kebapların yanındaki közlenmiş sebzeler bile özenle pişmişti, harikaydı.",
        "Garsonun tavırları o kadar asık suratlıydı ki neredeyse iştahımız kaçacaktı.",
        "Fiyat ve sunduğu performans açısından şehirdeki en başarılı mekan.",
        "Rezervasyonumuz olmasına rağmen adımız listede yoktu, uzun süre ayakta bekledik.",
        "Deniz mahsullü makarna tam tadındaydı, ellerinize sağlık.",
        "Servis o kadar yavaştı ki ana yemek gelene kadar ekmekle doymuştuk.",
        "Mekanın bahçesi ve çiçekli konsepti huzurlu bir atmosfer sunuyor.",
        "Musluk suyu tadında bir limonata ikram ettiler, hiç memnun kalmadık.",
        "Etin yanında gelen patates kızartması sıcacık ve harika baharatlanmıştı.",
        "Masaların temizliği pek iyi değildi, oturunca yüzeyin yapış yapış olduğunu fark ettik.",
        "Çocuk dostu bir işletme olması ailece bizim için büyük kolaylık sağladı.",
        "Hesap masaya geldiğinde şok geçirdik, menüdeki yazan fiyatlarla hiç uyuşmuyordu.",
        "Yemek sonrasında yediğimiz tatlının hafifliği kapanışı mükemmel yaptı.",
        "Siparişimiz tamamen yanlış geldi ve değiştirmelerini beklemek çok yordu.",
        "Hamburgerin ekmeği ev yapımı gibi çok yumuşak ve lezzetliydi.",
        "İçerideki havalandırma yetersizdi, tüm kıyafetlerimize yemek kokusu sindi.",
        "Mekanın merkezi konumu ve ulaşım kolaylığı çok büyük artı.",
        "Sipariş ettiğimiz balıklar maalesef aşırı pişmişti ve kuruydu.",
        "İşletme sahibi masaları tek tek dolaşıp hal hatır sordu, çok nazik bir davranıştı.",
        "Yemek porsiyonları fiyatına göre o kadar küçüktü ki doymadan kalktık.",
        "Kokteyller hem çok yaratıcıydı hem de tadı damağımızda kaldı.",
        "Garson siparişi yanlış aldıktan sonra bir de üstüne bizi suçladı.",
        "Günün menüsü hem doyurucu hem de bütçe dostuydu, çok sevdik.",
        "Klima o kadar güçlü çalışıyordu ki donduk, kapatmalarını söylediğimizde dikkate almadılar.",
        "Kahvelerinin aroması ve sunumu tam kaffe tutkunlarına göreydi.",
        "Ortamda inanılmaz bir uğultu ve gürültü vardı, birbirimizi duyamadık.",
        "Personel o kadar güler yüzlüydü ki kapıdan girdiğiniz an kendinizi iyi hissediyorsunuz.",
        "Yemeğin içinden saç teli çıktı, yetkiliye ilettiğimizde bile doğru dürüst özür dilemediler.",
        "Tatlıların kalorisine değdi diyebilirim, tek kelimeyle muazzamdı.",
        "Girişte bizi karşılayan görevli oldukça ilgisiz ve asık suratlıydı.",
        "Fiyat/performans oranı olarak bu bölgedeki en iyi mekan kesinlikle burası.",
        "İnternetten yaptığımız rezervasyon sistemde görünmüyormuş, kapıda mağdur olduk.",
        "Masanın üzerindeki taze çiçekler ve detaylar çok şık bir atmosfer yaratmıştı.",
        "Önceden sorduğumuz menü fiyatı kasada çok daha yüksek çıktı, sürpriz yaşadık."
    ],
    "label": [
        "positive", "negative", "positive", "negative", "positive",
        "positive", "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive",
        "positive", "positive", "positive", "negative", "negative",
        "positive", "positive", "positive", "negative", "positive",
        "negative", "positive", "positive", "positive", "positive",
        "negative", "positive", "positive", "negative", "negative",
        "negative", "positive", "positive", "positive", "positive",
        "positive", "positive", "positive", "positive", "negative",
        "negative", "positive", "positive", "positive", "negative",
        "positive", "negative", "positive", "negative", "negative",
        "positive", "negative", "negative", "positive", "positive",
        "positive", "negative", "positive", "negative", "positive",
        "negative","positive",
        "negative",
        "negative",
        "positive",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative"

    ]
    }

df= pd.DataFrame(data)
#metin temizleme ve preprocessing: tokenization padding label encoding train test split
tokenizer= Tokenizer()
tokenizer.fit_on_texts(df["text"])
"""

tokenizer.fit_on_texts(df["text"])

Sayısal Sözlük Yaratır: Bulduğu her kelimeye benzersiz bir tamsayı ID'si atar. 
Örneğin "lezzetli" kelimesine 15, "garson" kelimesine 42 gibi bir numara vererek
kendi içinde büyük bir kelime sözlüğü (word_index) oluşturur.
"""

sequences= tokenizer.texts_to_sequences(df["text"])

word_index = tokenizer.word_index
#padding process

maxlen= max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen = maxlen)
print(X.shape)

#label encoding
label_encoder= LabelEncoder()
y= label_encoder.fit_transform(df["label"])

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 42)
#metin temsili word embedding word2vec 
sentences = [text.split() for text in df["text"]]
word2vec_model= Word2Vec(sentences = sentences, vector_size=50, window=5, min_count=1, sg=0)

embedding_dim = 50
embedding_matrix = np.zeros((len(word_index)+1, embedding_dim))

for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i]= word2vec_model.wv[word]

#modeling build train test rnn model
model= Sequential()
#embedding
model.add(Embedding(input_dim = len(word_index)+1, output_dim = embedding_dim, weights= embedding_matrix, input_length= maxlen, trainable= False))
#RNN layer
model.add(SimpleRNN(50, return_sequences=False))

#output layer
model.add(Dense(1, activation="sigmoid"))

#model compile
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#train
model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test))

# evaluate rnn model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

#cumle classification

def sentence_classification(sentence):
    seq_token = tokenizer.texts_to_sequences([sentence])
    padded_seq = pad_sequences(seq_token, maxlen = maxlen)
    
    prediction = model.predict(padded_seq)
    
    predicted_class = (prediction > 0.5).astype(int)
    label = "positive" if predicted_class[0][0] == 1 else "negative"

    return label

sentence="Yemekleri beğenmedim."
result= sentence_classification(sentence)

print(f"Result: {result}")
#Result: negative

sentence2="Yemekler çok güzeldi, hayatımda yediğim en güzel yemekti."
result= sentence_classification(sentence2)

print(f"Result: {result}")
#Result: positive