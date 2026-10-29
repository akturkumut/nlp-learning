from transformers import pipeline

# "summarization" yerine "text-generation" kullanıyoruz

summarizer = pipeline("text-generation")

text = "The rapid advancement of artificial intelligence and digital technologies in the twenty-first century has fundamentally transformed the global economy, reshaping how industries operate, how businesses communicate, and how individuals interact with information on a daily basis. While these technological breakthroughs have undoubtedly unlocked unprecedented levels of efficiency, fostered innovation across diverse sectors such as healthcare and finance, and enabled seamless global connectivity, they have simultaneously introduced complex environmental and social challenges. Among the most pressing concerns is the exponential rise in energy consumption driven by massive data centers, complex algorithm training, and the continuous operation of digital infrastructure worldwide. As society becomes increasingly reliant on automated systems and cloud computing, the carbon footprint associated with maintaining these technologies threatens to undermine international efforts to combat climate change. Consequently, modern organizations and policymakers are now faced with a dual responsibility: they must continue to leverage cutting-edge digital tools to maintain competitive advantages, while actively developing sustainable strategies, renewable energy integrations, and ethical frameworks to mitigate the ecological and societal drawbacks of rapid technological growth."

# Modeme doğrudan bir komut veriyoruz
prompt = f"Summarize the following text concisely:\n\n{text}\n\nSummary:"

output = summarizer(
    prompt,
    max_new_tokens=45,  # text-generation'da max_length yerine max_new_tokens tercih edilir
    do_sample=False
)

# Üretilen metni ekrana yazdırıyoruz
print(output[0]["generated_text"])