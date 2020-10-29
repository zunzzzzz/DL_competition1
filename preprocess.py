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

    # author 
    if(text.find('author/') != -1):
        author_start = text.find('author/')
        author_end = author_start + 7
        while text[author_end] != '/': author_end += 1
        author = text[author_start+7:author_end-1]
    elif(text.find('<span class="author_name">') != -1): 
        author_start = text.find('<span class="author_name">') + 26
        author_end = text.find('</span>')
        author = text[author_start:author_end]
    elif(text.find('<span class="byline basic">') != -1): 
        author_start = text.find('<span class="byline basic">') + 27
        author_end = text.find('</span>')
        author = text[author_start:author_end]
    else:
        author = ''
    
    
    # channel 
    channel_start = text.find('channel="')
    channel_end = channel_start + 9
    while text[channel_end] != '"': channel_end += 1
    channel = text[channel_start+9:channel_end]
    # print(channel)
    # h2 h3
    h2 = ' '
    text_h2 = text
    while(text_h2.find('<h2>') > 0):
        h2_start = text_h2.find('<h2>')
        h2_end = h2_start + 4
        while text_h2[h2_end] != '<': h2_end += 1
        h2 = h2 + ' ' + text_h2[h2_start+4:h2_end-1]
        text_h2 = text_h2[h2_end+3:]
    h3 = ' '
    text_h3 = text
    while(text_h3.find('<h3>') > 0):
        h3_start = text_h3.find('<h3>')
        h3_end = h3_start + 4
        while text_h3[h3_end] != '<': h3_end += 1
        h3 = h3 + ' ' + text_h3[h3_start+4:h3_end-1]
        text_h3 = text_h3[h3_end+3:]

    # concatenate
    text = title + ' ' + topic + ' ' + month_day + ' ' + author
    
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
    stop.append('class')
    stop.append('title')
    my_stop = ['u', 'may', 'toward', 'also']
    stop.extend(my_stop)
    porter = PorterStemmer()
    return [porter.stem(w) for w in re.split('\s+', text.strip()) \
            if w not in stop]