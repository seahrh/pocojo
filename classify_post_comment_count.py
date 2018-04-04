import glob
import csv
import json
from os import path
from operator import itemgetter
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
from stringx.stringx import strip_punctuation

posts_glob_pattern = 'posts_txt/*.txt'
comments_glob_pattern = 'comments/*.json'

__random_state = 42
__folds = 3
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


def __grid_search(pipeline, param_grid, train, train_labels):
    grid = GridSearchCV(pipeline, cv=__folds, param_grid=param_grid)
    grid.fit(train, train_labels)
    print("Grid search\nBest: %f using %s" % (grid.best_score_,
                                              grid.best_params_))
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    params = grid.cv_results_['params']
    for mean, std, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, std, param))


def __pipeline(classifier, train, test, train_labels, test_labels, param_grid=None, is_tuning=False):
    print(f'#####  {classifier.__class__.__name__}  #####')
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(
            tokenizer=__tokenizer,
            preprocessor=__preprocessor,
            stop_words=__stopwords,
            min_df=0.02,
            sublinear_tf=True
        )),
        ('classifier', classifier)
    ])
    if is_tuning:
        if param_grid is None:
            param_grid = {}
        param_grid['tfidf__min_df'] = [1, 0.01, 0.05, 0.10]
        __grid_search(pipe, param_grid, train, train_labels)
        return
    scores = cross_val_score(pipe, train, train_labels, scoring='f1_macro', cv=__folds)
    print(f'Validation result\nf1_macro={np.median(scores)} (median)')
    pipe.fit(train, train_labels)
    preds = pipe.predict(test)
    report = classification_report(test_labels, preds)
    f1_macro = f1_score(test_labels, preds, average='macro')
    f1_micro = f1_score(test_labels, preds, average='micro')
    print(f'Test result\nf1_macro={f1_macro}\nf1_micro={f1_micro}\n{report}')


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
        posts, labels, test_size=0.1, random_state=__random_state
    )
    is_tuning = True
    __pipeline(MultinomialNB(), train, test, train_labels, test_labels, is_tuning=is_tuning)
    __pipeline(LinearSVC(), train, test, train_labels, test_labels, is_tuning=is_tuning)
    __pipeline(RandomForestClassifier(
        n_jobs=-1,
        random_state=__random_state
    ), train, test, train_labels, test_labels, is_tuning=is_tuning)
    __pipeline(GradientBoostingClassifier(
        random_state=__random_state
    ), train, test, train_labels, test_labels, is_tuning=is_tuning)


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


main()
