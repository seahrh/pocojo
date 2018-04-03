import glob
import csv
from os import path
from operator import itemgetter
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from scipy.sparse import save_npz
from stringx.stringx import strip_punctuation

posts_glob_pattern = 'posts_txt/*.txt'

__stemmer = PorterStemmer()
__stopwords = set(stopwords.words('english'))


def __preprocessor(text):
    s = text.lower()
    s = s.replace('\r', ' ').replace('\n', ' ').strip()
    s = strip_punctuation(s)
    return s


def __stem_tokens(tokens, stemmer):
    stemmed = []
    for t in tokens:
        stemmed.append(stemmer.stem(t))
    return stemmed


def __tokenizer(text):
    tokens = nltk.word_tokenize(text)
    stems = __stem_tokens(tokens, __stemmer)
    return stems


def __pipeline(clf_type):
    pipe = Pipeline([
        ('vect', TfidfVectorizer(
            tokenizer=__tokenizer,
            preprocessor=__preprocessor,
            stop_words=__stopwords,
            min_df=0.1,
            sublinear_tf=True
        )),
        ('clf', clf_type()),
    ])
    return pipe


def foo():
    pid_to_tokens = OrderedDict()
    paths = glob.glob(posts_glob_pattern)
    for p in paths:
        pid = path.basename(p)[1:-4]
        with open(p, 'rt') as f:
            s = f.read()
            # s = s.lower().replace('\r', ' ').replace('\n', ' ').strip()
            # s = strip_punctuation(s)
            # print(f'pid={repr(pid)}\ns={repr(s)}')
            pid_to_tokens[pid] = s

    tf_idf = TfidfVectorizer(
        tokenizer=__tokenizer,
        preprocessor=__preprocessor,
        stop_words=__stopwords,
        min_df=0.1,
        sublinear_tf=True
    )
    ws = tf_idf.fit_transform(pid_to_tokens.values())
    terms = tf_idf.get_feature_names()
    print(f'ws.shape={repr(ws.shape)}')
    idf = tf_idf.idf_
    term_to_idf = sorted(zip(terms, idf), key=itemgetter(1))
    print(f'term_to_idf.len={repr(len(term_to_idf))}')
    f = csv.writer(open("tmp/idf.csv", "wt"))
    for term, idf in term_to_idf:
        f.writerow([term, idf])
    save_npz('tmp/ws', ws)


foo()
