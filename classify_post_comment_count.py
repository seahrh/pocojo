import glob
from os import path
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from stringx.stringx import strip_punctuation

posts_glob_pattern = 'posts_txt/*.txt'

__stemmer = PorterStemmer()
__stopwords = set(stopwords.words('english'))


def stem_tokens(tokens, stemmer):
    stemmed = []
    for t in tokens:
        stemmed.append(stemmer.stem(t))
    return stemmed


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, __stemmer)
    return stems


pid_to_tokens = OrderedDict()
paths = glob.glob(posts_glob_pattern)[:5]
for p in paths:
    pid = path.basename(p)[1:-4]
    with open(p, 'rt') as f:
        s = f.read().lower()
        s = s.replace('\r', ' ').replace('\n', ' ').strip()
        s = strip_punctuation(s)
        # print(f'pid={repr(pid)}\ns={repr(s)}')
        pid_to_tokens[pid] = s

tf_idf = TfidfVectorizer(tokenizer=tokenize, stop_words=__stopwords)
ws = tf_idf.fit_transform(pid_to_tokens.values())
print(f'ws.shape={repr(ws.shape)}\n{repr(ws)}')
