import csv
from operator import itemgetter
from pprint import pprint
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, classification_report, r2_score, median_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

from sklearnpd.sklearnpd import TextExtractor, TransformPipeline
from stringx.stringx import strip_punctuation, is_number
from timex.timex import Timer, seconds_to_hhmmss

__posts_glob_pattern = 'posts_txt/*.txt'
__comments_glob_pattern = 'comments/*.json'
__in_file_path = 'tmp/data.tsv'
__in_file_separator = '\t'
__idf_file_path = 'tmp/idf.txt'
__coefs_file_path = 'tmp/coefs.txt'
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


def __grid_search(pipeline, param_grid, train, train_labels, scoring):
    print(f'param_grid={repr(param_grid)}\nTuning...')
    grid = GridSearchCV(pipeline, cv=__folds, param_grid=param_grid, scoring=scoring)
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


def __train(pipeline, train, train_y, step_features='features', step_model='model'):
    print('Training...')
    timer = Timer()
    timer.start()
    pipeline.fit(train, train_y)
    features = pipeline.named_steps[step_features]
    vectorizer = features.transformer_list[0][1].named_steps['vector']
    __idf(vectorizer, __idf_file_path)
    model = pipeline.named_steps[step_model]
    __coefs(features, model, __coefs_file_path)
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


def __coefs(features, model, file_path):
    feature_names = features.get_feature_names()
    coefs = model.coef_[0]
    intercept = model.intercept_
    print(f'features len={len(feature_names)}\ncoefs len={len(coefs)}\nintercept={intercept}')
    ranked_features = []
    for i in coefs.argsort()[::-1]:
        ranked_features.append((feature_names[i], coefs[i]))
    with open(file_path, 'wt') as out:
        pprint(ranked_features, stream=out)


def __pipeline(classifier, train, test, train_y, test_y, scoring, task='train'):
    cls_name = classifier.__class__.__name__
    print(f'#####  {cls_name}  #####')
    timer = Timer()
    timer.start()
    pipe = Pipeline([
        ('features', FeatureUnion([
            ('tfidf', TransformPipeline([
                ('extract', TextExtractor(col='text')),
                ('vector', TfidfVectorizer(
                    tokenizer=__tokenizer,
                    preprocessor=__preprocessor,
                    stop_words=__stopwords,
                    min_df=0.01,
                    sublinear_tf=True
                )),
                ('scale', MaxAbsScaler())
            ]))
        ])),
        ('model', classifier)
    ])
    # print(f'pipeline steps={repr(pipe.steps)}')
    if task == 'test':
        __train(pipe, train, train_y)
        __test(pipe, test, test_y)
    elif task == 'train':
        __train(pipe, train, train_y)
    elif task == 'tune':
        param_grid = {
            'features__tfidf__scale': [None, MaxAbsScaler()]
            #'features__tfidf__vector__min_df': [1, 10, 100]
        }
        __grid_search(pipe, param_grid, train, train_y, scoring=scoring)
    elif task == 'validate':
        __validate(pipe, train, train_y, scoring)
    else:
        raise ValueError(f'Invalid value: task={task}')
    timer.stop()
    print(f'__pipeline: {cls_name},{task} took {seconds_to_hhmmss(timer.elapsed)}')


def __multinomial_nb(train, test, train_labels, test_labels, task):
    __pipeline(MultinomialNB(), train, test, train_labels, test_labels, scoring='f1_macro',
               task=task)


def __linear_svc(train, test, train_labels, test_labels, task):
    __pipeline(LinearSVC(), train, test, train_labels, test_labels, scoring='f1_macro',
               task=task)


def __random_forest(train, test, train_labels, test_labels, task):
    __pipeline(RandomForestClassifier(
        n_jobs=-1,
        random_state=__random_state
    ), train, test, train_labels, test_labels, scoring='f1_macro', task=task)


def __gradient_boosting(train, test, train_labels, test_labels, task):
    __pipeline(GradientBoostingClassifier(
        random_state=__random_state
    ), train, test, train_labels, test_labels, scoring='f1_macro', task=task)


def __main(task):
    df = pd.read_csv(__in_file_path, sep=__in_file_separator)
    ys = df.loc[:, ['comment_count']]
    # Exclude the first 2 columns: row index, label
    xs = df.iloc[:, 2:]
    print(f'ys shape={ys.shape}, columns={ys.columns.values.tolist()}')
    print(f'xs shape={xs.shape}, head={xs.columns.values.tolist()[:10]}')
    train, test, train_y, test_y = train_test_split(
        xs, ys, test_size=0.1, random_state=__random_state
    )
    __pipeline(Ridge(
        alpha=1.0
    ), train, test, train_y, test_y, scoring='r2', task=task)


def __idf(vectorizer, file_path):
    idfs = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    print(f'idfs len={len(idfs)}')
    with open(file_path, 'wt') as out:
        pprint(idfs, stream=out)


if __name__ == '__main__':
    import sys

    __task = sys.argv[1].lower()
    __main(__task)
