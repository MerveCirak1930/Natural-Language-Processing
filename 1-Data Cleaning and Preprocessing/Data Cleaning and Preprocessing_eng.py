
################################################################3
                             ### STEMMING ingilizce###
import nltk
#nltk.download("wordnet")

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

# ornek kelimeler
kelimeler = ["running","runner","went","runs"] #['run', 'runner', 'went', 'run']

kokler = [stemmer.stem(k) for k in kelimeler]
print("Kelimelerin kokleri:",kokler)



                            ### LEMMATİZATİON ingilizce###
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# ornek kelimeler
kelimeler = ["running","runner","went","runs"]  #['run', 'runner', 'go', 'run']

lemmas = [lemmatizer.lemmatize(k,pos="v") for k in kelimeler]

print("Lemma lar:",lemmas)





                            ## STOP WORDS ###

import nltk
from nltk.corpus import stopwords


# Eğer stopwords veri seti eksikse indir
nltk.download("stopwords")


# Stop-words listesini yükle
stop_words = set(stopwords.words("english"))

# Text example
text ="This is an example of removing stop words from a text document"

# Filtered text (lack of stop words)
filtered_words = [word for word in text.split() if word.lower() not in stop_words]   

print("Original text:",text)
print("Filtered words:",filtered_words)

print("Filtered text:", " ".join(filtered_words))


# CIKTI                            

# Original text: This is an example of removing stop words from a text document
# filtered words: ['example', 'removing', 'stop', 'words', 'text', 'document']
# Stop-word'ler temizlenmiş hali: example removing stop words text document