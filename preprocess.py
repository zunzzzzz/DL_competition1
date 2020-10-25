import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def tokenizer_stem_nostop(text):
    nltk.download('stopwords')
    stop = stopwords.words('english')
    porter = PorterStemmer()
    return [porter.stem(w) for w in re.split('\s+', text.strip()) \
            if w not in stop and re.match('[a-zA-Z]+', w)]