from pprint import pprint

import nltk
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.metrics import f1_score, classification_report, r2_score, median_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MaxAbsScaler

from sklearnpd.sklearnpd import ColumnExtractor, TransformPipeline, PrefixColumnExtractor,\
    TransformLatentDirichletAllocation
from stopwords import stopwords
from stringx.stringx import strip_punctuation, is_number
from timex.timex import Timer, seconds_to_hhmmss

__posts_glob_pattern = 'posts_txt/*.txt'
__comments_glob_pattern = 'comments/*.json'
__in_file_path = 'tmp/data.tsv'
__in_file_separator = '\t'
__idf_file_path = 'tmp/idf.txt'
__coefs_file_path = 'tmp/coefs.txt'
__topic_term_file_path = 'tmp/topic_term.txt'
__is_tuning = False
__random_state = 42
__folds = 3
__stemmer = PorterStemmer()
__metric_median_absolute_error = 'neg_median_absolute_error'


def __preprocessor(text):
    """
    Happens before tokenizer.
    :param text: text
    :return: text
    """
    s = text.lower()
    s = s.replace('\r', ' ').replace('\n', ' ').strip()
    s = strip_punctuation(s)
    return s


def __tokenizer(text):
    tokens = nltk.word_tokenize(text)
    stems = [__stemmer.stem(t) for t in tokens]
    tokens = [t for t in stems if not is_number(t)]
    return tokens


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
    lda = features.transformer_list[1][1].named_steps['lda']
    __topic_term(lda, __topic_term_file_path)
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


def __topic_term(transformer, file_path):
    tt = transformer.components_
    print(f'topic_to_term shape={np.shape(tt)}, saved {repr(file_path)}')
    with open(file_path, 'wt') as out:
        pprint(tt, stream=out)


def __coefs(features, model, file_path):
    feature_names = features.get_feature_names()
    coefs = np.ravel(model.coef_)
    intercept = model.intercept_
    print(f'''features len={len(feature_names)}
coefs len={len(coefs)}, saved {repr(file_path)}
intercept={intercept}''')
    ranked_features = []
    for i in coefs.argsort()[::-1]:
        ranked_features.append((feature_names[i], coefs[i]))
    with open(file_path, 'wt') as out:
        pprint(ranked_features, stream=out)


def __pipeline(classifier, train, test, train_y, test_y, scoring, task='train'):
    cls_name = classifier.__class__.__name__
    print(f'#####  {cls_name}:{task}  #####')
    timer = Timer()
    timer.start()
    pipe = Pipeline([
        ('features', FeatureUnion([
            ('tfidf', TransformPipeline([
                ('extract', ColumnExtractor(col='text', as_type=str)),
                ('vector', TfidfVectorizer(
                    analyzer='word',
                    tokenizer=__tokenizer,
                    preprocessor=__preprocessor,
                    stop_words=stopwords(),
                    min_df=10,
                    sublinear_tf=True
                ))
            ])),
            ('topic', TransformPipeline([
                ('extract', ColumnExtractor(col='text', as_type=str)),
                ('vector', CountVectorizer(
                    analyzer='word',
                    tokenizer=__tokenizer,
                    preprocessor=__preprocessor,
                    stop_words=stopwords(),
                    min_df=10
                )),
                ('lda', TransformLatentDirichletAllocation(
                    n_components=8,
                    max_iter=3,
                    learning_method='online',
                    learning_offset=10.,
                    n_jobs=4,
                    random_state=__random_state
                ))
            ])),
            ('author', PrefixColumnExtractor(prefix='author_', as_type=int)),
            ('token_count', TransformPipeline([
                ('extract', ColumnExtractor(col='token_count', as_type=int, as_matrix=True)),
                ('scale', MaxAbsScaler())
            ])),
            ('token_length_mean', TransformPipeline([
                ('extract', ColumnExtractor(col='token_length_mean', as_type=float, as_matrix=True)),
                ('scale', MaxAbsScaler())
            ])),
            ('char_count', TransformPipeline([
                ('extract', ColumnExtractor(col='char_count', as_type=int, as_matrix=True)),
                ('scale', MaxAbsScaler())
            ])),
            ('digit_char_ratio', ColumnExtractor(col='digit_char_ratio', as_type=float, as_matrix=True)),
            ('alpha_char_ratio', ColumnExtractor(col='alpha_char_ratio', as_type=float, as_matrix=True)),
            ('upper_char_ratio', ColumnExtractor(col='upper_char_ratio', as_type=float, as_matrix=True)),
            ('space_char_ratio', ColumnExtractor(col='space_char_ratio', as_type=float, as_matrix=True)),
            ('punctuation_char_ratio', ColumnExtractor(col='punctuation_char_ratio', as_type=float, as_matrix=True))
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
            # 'features__token_length_mean__scale': [None, MaxAbsScaler()]
            # 'features__tfidf__vector__min_df': [1, 10, 100]
            # 'features__topic__lda__n_components': [8, 16, 32]
            # 'features__topic__lda__max_iter': [3, 6, 10]
            # 'model__penalty': ['l2', 'l1', 'elasticnet']
            # 'model__alpha': [0.0001, 0.001, 0.01]
            'model__loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
        }
        __grid_search(pipe, param_grid, train, train_y, scoring=scoring)
    elif task == 'validate':
        __validate(pipe, train, train_y, scoring)
    else:
        raise ValueError(f'Invalid value: task={task}')
    timer.stop()
    print(f'__pipeline {cls_name}:{task} took {seconds_to_hhmmss(timer.elapsed)}')


def __multinomial_nb(train, test, train_labels, test_labels, task, scoring='f1_macro'):
    __pipeline(MultinomialNB(), train, test, train_labels, test_labels, scoring=scoring,
               task=task)


def __linear_svc(train, test, train_labels, test_labels, task, scoring='f1_macro'):
    __pipeline(LinearSVC(), train, test, train_labels, test_labels, scoring=scoring,
               task=task)


def __random_forest(train, test, train_labels, test_labels, task, scoring='f1_macro'):
    __pipeline(RandomForestClassifier(
        n_jobs=-1,
        random_state=__random_state
    ), train, test, train_labels, test_labels, scoring=scoring, task=task)


def __gradient_boosting(train, test, train_labels, test_labels, task, scoring='f1_macro'):
    __pipeline(GradientBoostingClassifier(
        random_state=__random_state
    ), train, test, train_labels, test_labels, scoring=scoring, task=task)


def __ridge(train, test, train_y, test_y, task, scoring='r2'):
    __pipeline(Ridge(
        alpha=1.0
    ), train, test, train_y, test_y, scoring=scoring, task=task)


def __sgd_regressor(train, test, train_y, test_y, task, scoring='r2'):
    __pipeline(SGDRegressor(
        penalty='l2',
        max_iter=1000,
        random_state=__random_state
    ), train, test, train_y, test_y, scoring=scoring, task=task)


def __main(model, task):
    df = pd.read_csv(__in_file_path, sep=__in_file_separator)
    ys = df.loc[:, ['comment_count']]
    # Exclude the first 2 columns: row index, label
    xs = df.iloc[:, 2:]
    print(f'ys shape={ys.shape}, columns={ys.columns.values.tolist()}')
    print(f'xs shape={xs.shape}, head={xs.columns.values.tolist()[:10]}')
    train, test, train_y, test_y = train_test_split(
        xs, ys, test_size=0.1, random_state=__random_state
    )
    if model == 'ridge':
        __ridge(train, test, train_y, test_y, task)
    elif model == 'sgd':
        __sgd_regressor(train, test, train_y, test_y, task)
    else:
        raise ValueError(f'Invalid model={model}')


def __idf(vectorizer, file_path):
    idfs = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    print(f'idfs len={len(idfs)}, saved {repr(file_path)}')
    with open(file_path, 'wt') as out:
        pprint(idfs, stream=out)


if __name__ == '__main__':
    import sys

    __model = sys.argv[1].lower()
    __task = sys.argv[2].lower()
    __main(__model, __task)
