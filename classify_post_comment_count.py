import glob
import csv
import json
from os import path
from operator import itemgetter
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import metrics
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from scipy.sparse import save_npz
from stringx.stringx import strip_punctuation

posts_glob_pattern = 'posts_txt/*.txt'
comments_glob_pattern = 'comments/*.json'

__stemmer = PorterStemmer()
__stopwords = set(stopwords.words('english'))


def comment_count(jo):
    comments = jo['comments']
    n = 0
    for i, comment in enumerate(comments):
        n += 1
        for _ in comment['children']:
            n += 1
    return n


def __label(n_comment):
    if n_comment < 0:
        raise ValueError('comment count must not be less than 0')
    if 0 <= n_comment <= 2:
        return 'lo'
    if 3 <= n_comment <= 5:
        return 'mi'
    return 'hi'


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


def __pipeline(clf):
    pipe = Pipeline([
        ('vect', TfidfVectorizer(
            tokenizer=__tokenizer,
            preprocessor=__preprocessor,
            stop_words=__stopwords,
            min_df=0.02,
            sublinear_tf=True
        )),
        ('clf', clf),
    ])
    return pipe


def main():
    posts = list()
    paths = sorted(glob.glob(posts_glob_pattern))
    for p in paths:
        with open(p, 'rt') as f:
            s = f.read()
            posts.append(s)
    # print(f'posts={repr(posts[:10])}')
    labels = list()
    paths = sorted(glob.glob(comments_glob_pattern))
    for p in paths:
        with open(p, 'rt') as f:
            _jo = json.load(f)
            n_comment = comment_count(_jo)
            labels.append(__label(n_comment))
    # print(f'labels={repr(labels[:10])}')
    train, test, train_labels, test_labels = train_test_split(
        posts, labels, test_size=0.1, random_state=42
    )
    pipe = __pipeline(MultinomialNB())
    pipe.fit(train, train_labels)
    preds = pipe.predict(test)
    # print report
    report = metrics.classification_report(test_labels, preds)
    print(f'MultinomialNB\n{report}')
    pipe = __pipeline(LinearSVC())
    pipe.fit(train, train_labels)
    preds = pipe.predict(test)
    # print report
    report = metrics.classification_report(test_labels, preds)
    print(f'LinearSVC\n{report}')
    pipe = __pipeline(RandomForestClassifier(
        n_jobs=-1,
        random_state=42
    ))
    pipe.fit(train, train_labels)
    preds = pipe.predict(test)
    # print report
    report = metrics.classification_report(test_labels, preds)
    print(f'RandomForestClassifier\n{report}')
    pipe = __pipeline(GradientBoostingClassifier(
        random_state=42
    ))
    pipe.fit(train, train_labels)
    preds = pipe.predict(test)
    # print report
    report = metrics.classification_report(test_labels, preds)
    print(f'GradientBoostingClassifier\n{report}')


def foo():
    pid_to_tokens = OrderedDict()
    paths = glob.glob(posts_glob_pattern)
    for p in paths:
        pid = path.basename(p)[1:-4]
        with open(p, 'rt') as f:
            s = f.read()
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


main()
