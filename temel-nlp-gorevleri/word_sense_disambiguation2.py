#import lib
from pywsd.lesk import simple_lesk, adapted_lesk, cosine_lesk

sentences = [
    "I was in central bank in London for deposit money",
    "The river bank was flooded after the heavy rain"]

word = "bank"

for s in sentences:
    print(f"Cumle: {s}")
    sense_simple_lesk= simple_lesk(s, word)
    print(f"Sense simple: {sense_simple_lesk.definition()}")
    
    sense_adapted_lesk= adapted_lesk(s, word)
    print(f"Sense adapted: {sense_adapted_lesk.definition()}")
    
    sense_cosine_lesk = cosine_lesk(s, word)
    print(f"Sense cosine: {sense_cosine_lesk.definition()}")
    
"""
Cumle: I was in central bank in London for deposit money
Sense simple: a financial institution that accepts deposits and channels the money into lending activities
Sense adapted: a financial institution that accepts deposits and channels the money into lending activities
Sense cosine: enclose with a bank

Cumle: The river bank was flooded after the heavy rain
Sense simple: sloping land (especially the slope beside a body of water)
Sense adapted: sloping land (especially the slope beside a body of water)
Sense cosine: a supply or stock held in reserve for future use (especially in emergencies)
"""