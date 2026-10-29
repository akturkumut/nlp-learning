from google import genai

client = genai.Client(api_key= "api_key")



if __name__ == "__main__":
    
    # Sohbet oturumunu döngünün DIŞINDA başlatıyoruz ki hafızasını korusun
    chat = client.chats.create(model="gemini-3.8-flash")
    
    print("Chatbot başlatıldı. Çıkmak için 'q' veya 'exit' yazın.")
    
    while True:
        user_input = input("Mesajınız : ")
        
        if user_input.lower() in ["exit", "q"]:
            print("Konuşma sonlandı.")
            break
            
        # Doğrudan mevcut chat oturumu üzerinden mesaj gönderiyoruz
        response = chat.send_message(user_input)
        
        print(f"Chatbot: {response.text.strip()}\n")

