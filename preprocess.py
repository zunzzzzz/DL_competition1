import re
from bs4 import BeautifulSoup
import numpy as np
def preprocessor(text):
    # store useful information

    # title
    title_start = text.find('<h1 class="title">')
    title_end = text.find('</h1>')
    title = text[title_start:title_end]
    # topic
    topic_idx = text.find('Topics:')
    topic = text[topic_idx+7:]

    # month and day
    time_idx = text.find('<time datetime=')
    day = text[time_idx+16:time_idx+19]
    month = text[time_idx+24:time_idx+27]
    month_day = month + ' ' + day
    # print(month_day)
    # concatenate
    text = title + topic + ' ' + month_day
    # remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()
    
    # regex for matching emoticons, keep emoticons, ex: :), :-P, :-D
    r = '(?::|;|=|X)(?:-)?(?:\)|\(|D|P)'
    emoticons = re.findall(r, text)
    text = re.sub(r, '', text)
    
    # convert to lowercase and append all emoticons behind (with space in between)
    # replace('-','') removes nose of emoticons
    text = re.sub('[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-','')
    return text

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def tokenizer_stem_nostop(text):
    nltk.download('stopwords', quiet=True)
    stop = stopwords.words('english')
    my_stop = ['u', 'may', 'toward', 'also']
    stop.extend(my_stop)
    porter = PorterStemmer()
    return [porter.stem(w) for w in re.split('\s+', text.strip()) \
            if w not in stop]