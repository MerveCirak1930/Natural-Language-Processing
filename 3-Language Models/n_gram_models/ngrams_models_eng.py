
from collections import Counter

import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

# Sample data set
corpus = [
    "I love you",
    "I love apple",
    "I love programming",
    "You love me",
    "She loves apple",
    "They love you",
    "I love you and you love me"
]

# Tokenization
tokens =[word_tokenize(sentence.lower()) for sentence in corpus]
#print(tokens) # [['i', 'love', 'you'], ['i', 'love', 'apple'], ['i', 'love', 'programming'], ['you', 'love', 'me'], ['she', 'loves', 'apple'], ['they', 'love', 'you'], ['i', 'love', 'you', 'and', 'you', 'love', 'me']]

# n-gram : n=2
bigrams = []
for token_list in tokens:
    bigrams.extend(list(ngrams(token_list, 2)))

#print(bigrams) #[('i', 'love'), ('love', 'you'), ('i', 'love'), ('love', 'apple'), ('i', 'love'), ('love', 'programming'), ('you', 'love'), ('love', 'me'), ('she', 'loves'), ('loves', 'apple'), ('they', 'love'), ('love', 'you'), ('i', 'love'), ('love', 'you'), ('you', 'and'), ('and', 'you'), ('you', 'love'), ('love', 'me')]

# bigram frequency
bigram_freq = Counter(bigrams)

# Sonuçları yazdırma
print("\nBigram Frekansları:")
for bigram, freq in bigram_freq.items():
    print(f"{bigram}: {freq}")

# n-gram : n=3
trigrams = []
for token_list in tokens:
    trigrams.extend(list(ngrams(token_list, 3)))

print(trigrams)  

# trigram frequency
trigram_freq = Counter(trigrams)

# Sonuçları yazdırma
print("\nTrigram Frekansları:")
for trigram, freq in trigram_freq.items():
    print(f"{trigram}: {freq}") 


# "I love" bigramından sonra "you" veya "apple" gelme olasılıklarını hesapla ########
bigram = ("i","love")

prob_you = trigram_freq[("i","love","you")] / bigram_freq[bigram]
prob_apple = trigram_freq[("i","love","apple")] / bigram_freq[bigram]

print("you kelimesinin gelme olasılığı:",prob_you)
print("apple kelimesinin gelme olasılığı:",prob_apple)