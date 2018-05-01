import csv
from operator import itemgetter

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report, r2_score, median_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge

from sklearnpd.sklearnpd import TextExtractor
from stringx.stringx import strip_punctuation, is_number
from timex.timex import Timer, seconds_to_hhmmss

posts_glob_pattern = 'posts_txt/*.txt'
comments_glob_pattern = 'comments/*.json'
__in_file_path = 'tmp/data.tsv'
__in_file_separator = '\t'
__is_tuning = False
__random_state = 42
__folds = 3
__stemmer = PorterStemmer()
__stopwords = set(stopwords.words('english'))


def num_words(s):
    return len(s.split())


def ave_word_length(s):
    return np.mean([len(w) for w in s.split()])


def __preprocessor(text):
    s = text.lower()
    s = s.replace('\r', ' ').replace('\n', ' ').strip()
    s = strip_punctuation(s)
    return s


def __tokenizer(text):
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if not is_number(t)]
    stems = [__stemmer.stem(t) for t in tokens]
    return stems


def __grid_search(pipeline, param_grid, train, train_labels):
    grid = GridSearchCV(pipeline, cv=__folds, param_grid=param_grid, scoring='f1_macro')
    grid.fit(train, train_labels)
    print("Grid search\nBest: %f using %s" % (grid.best_score_,
                                              grid.best_params_))
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    params = grid.cv_results_['params']
    for mean, std, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, std, param))


def __test_classification(pipeline, test, test_y):
    print('Testing...')
    preds = pipeline.predict(test)
    report = classification_report(test_y, preds)
    f1_mac = f1_score(test_y, preds, average='macro')
    f1_mic = f1_score(test_y, preds, average='micro')
    print(f'Test result\nf1_macro={f1_mac}\nf1_micro={f1_mic}\n{report}')


def __test(pipeline, test, test_y):
    print('Testing...')
    preds = pipeline.predict(test)
    r2 = r2_score(test_y, preds)
    mae = median_absolute_error(test_y, preds)
    print(f'Test result: r2={r2}, mae={mae}')


def __train(pipeline, train, train_y):
    print('Training...')
    timer = Timer()
    timer.start()
    pipeline.fit(train, train_y)
    timer.stop()
    print(f'__train took {seconds_to_hhmmss(timer.elapsed)}')


def __validate(pipeline, train, train_y, scoring):
    print('Validating...')
    timer = Timer()
    timer.start()
    scores = cross_val_score(pipeline, train, train_y, scoring=scoring, cv=__folds)
    print(f'Validation result: {scoring}={np.median(scores)} (median)')
    timer.stop()
    print(f'__validate took {seconds_to_hhmmss(timer.elapsed)}')


def __pipeline(classifier, train, test, train_y, test_y, scoring,
               param_grid=None, is_tuning=False):
    print(f'#####  {classifier.__class__.__name__}  #####')
    timer = Timer()
    timer.start()
    pipe = Pipeline([
        ('features', FeatureUnion([
            ('tfidf', Pipeline([
                ('extract', TextExtractor(col='text')),
                ('vector', TfidfVectorizer(
                    tokenizer=__tokenizer,
                    preprocessor=__preprocessor,
                    stop_words=__stopwords,
                    min_df=0.01,
                    sublinear_tf=True
                ))
            ]))
        ])),
        ('model', classifier)
    ])
    # print(f'pipeline steps={repr(pipe.steps)}')
    if is_tuning:
        if param_grid is None:
            param_grid = {}
        param_grid['features__tfidf__vector__min_df'] = [1, 0.01, 0.05]
        __grid_search(pipe, param_grid, train, train_y)
    else:
        #__validate(pipe, train, train_y, scoring)
        __train(pipe, train, train_y)
        fs = pipe.named_steps['features'].transformer_list[0][1].named_steps['vector'].get_feature_names()
        print(f'fs len={len(fs)}, some={fs[::10]}')
        coefs = pipe.named_steps['model'].coef_
        intercept = pipe.named_steps['model'].intercept_
        print(f'coefs shape={np.shape(coefs)}, some={coefs[::10]}, intercept={intercept}')
        __test(pipe, test, test_y)
    timer.stop()
    print(f'__pipeline took {seconds_to_hhmmss(timer.elapsed)}')


def __multinomial_nb(train, test, train_labels, test_labels):
    __pipeline(MultinomialNB(), train, test, train_labels, test_labels, scoring='f1_macro',
               is_tuning=__is_tuning)


def __linear_svc(train, test, train_labels, test_labels):
    __pipeline(LinearSVC(), train, test, train_labels, test_labels, scoring='f1_macro',
               is_tuning=__is_tuning)


def __random_forest(train, test, train_labels, test_labels):
    __pipeline(RandomForestClassifier(
        n_jobs=-1,
        random_state=__random_state
    ), train, test, train_labels, test_labels, scoring='f1_macro', is_tuning=__is_tuning)


def __gradient_boosting(train, test, train_labels, test_labels):
    __pipeline(GradientBoostingClassifier(
        random_state=__random_state
    ), train, test, train_labels, test_labels, scoring='f1_macro', is_tuning=__is_tuning)


def __main():
    df = pd.read_csv(__in_file_path, sep=__in_file_separator)
    ys = df.loc[:, ['comment_count']]
    # Exclude the first 2 columns: row index, label
    xs = df.iloc[:, 2:]
    print(f'ys columns={ys.columns.values.tolist()}, shape={ys.shape}')
    print(f'xs top_columns={xs.columns.values.tolist()[:10]}, shape={xs.shape}')
    train, test, train_y, test_y = train_test_split(
        xs, ys, test_size=0.1, random_state=__random_state
    )
    __pipeline(Ridge(
        alpha=1.0
    ), train, test, train_y, test_y, scoring='r2', is_tuning=__is_tuning)


def __save_idf(tf_idf):
    terms = tf_idf.get_feature_names()
    idf = tf_idf.idf_
    term_to_idf = sorted(zip(terms, idf), key=itemgetter(1))
    print(f'term_to_idf.len={repr(len(term_to_idf))}')
    f = csv.writer(open("tmp/idf.csv", "wt"))
    for term, idf in term_to_idf:
        f.writerow([term, idf])


if __name__ == '__main__':
    import sys

    __is_tuning = sys.argv[1].lower() == 'true'
    __main()
