# POCOJO

POsts-COmments-JObs dataset from a news outlet based in Singapore.

## Data

Size
* 32,188 posts (worked around the page limit in the posts api)
* Downloaded comments for each post

## Predict comment-count

Changed from classification task to regression, where the target variable is the comment count of the post.

### Results

Best result: Stochastic gradient descent (SGD) regression, with R2 score of 0.34 and median absolute error of 2.2.

### Splitting

* Test set holdout at 10% of posts
* Reproducible random split, because random seed is fixed

### Preprocessing

* Parse json to extract fields
* Remove HTML tags
* Convert HTML entities to ASCII
* Normalize unicode/accented characters: change string encoding to ASCII
* Count comments by post
* Extract features (e.g. text, author name) from json to pandas dataframe
* Serialize pandas dataframe as TSV file (sole data file to be used for ML)

### Feature Engineering

In all, more than 20K features were created.
* tf-idf weights
    * Lowercase
    * Remove stopwords (hand picked + [sklearn's base set](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/stop_words.py))
    * Remove punctuation
    * Stemming (Porter)
    * Filtering: document frequency thresholding
    * All terms and their [IDF scores](result/idf.txt)
* Topic weights
    * How each post is contributing to a range of unlabelled topics - a post has a score for a given topic, this is known as the topic weight
    * Number of topics set as 8 (after parameter tuning) 
    * Model: [Latent Dirichlet Allocation (LDA)](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)
* Token count
* Average token length
* Author name (one-hot encoded)
* Character level features (normalised by total character count) 
    * Total character count
    * Digit count
    * Letter count
    * Uppercase character count
    * Whitespace character count
    * Punctuation character count
    
All features are listed in [coefs.txt](result/coefs.txt), in descending order of the coefficient value. Highlights:
* tf-idf weights alone account for a R2 score of 0.24
* Expectedly, some author features have high coefficients
* Topic features do not have as large coefficients as hoped, need for more tuning
* Many features with near-zero coefficient, need for feature selection

### Models

* [Stochastic Gradient Descent](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)
* [Ridge Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)

### Parameter tuning

Tuned the following parameters with [Grid search](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html). Scores shown are R2.

```
Best: 0.342537 using {'model__alpha': 0.0001}
0.342537 (0.060945) with: {'model__alpha': 0.0001}
0.297158 (0.072006) with: {'model__alpha': 0.001}
0.123099 (0.038804) with: {'model__alpha': 0.01}
__pipeline SGDRegressor:tune took 03:48:20

Best: 0.342537 using {'model__penalty': 'elasticnet'}
0.342472 (0.061780) with: {'model__penalty': 'l2'}
0.341036 (0.055095) with: {'model__penalty': 'l1'}
0.342537 (0.060945) with: {'model__penalty': 'elasticnet'}
__pipeline SGDRegressor:tune took 03:57:42

Best: 0.342472 using {'features__topic__lda__max_iter': 3}
0.342472 (0.061780) with: {'features__topic__lda__max_iter': 3}
0.342433 (0.061793) with: {'features__topic__lda__max_iter': 6}
0.342428 (0.061806) with: {'features__topic__lda__max_iter': 10}
__pipeline SGDRegressor:tune took 04:07:21

Best: 0.342428 using {'features__topic__lda__n_components': 8}
0.342428 (0.061806) with: {'features__topic__lda__n_components': 8}
0.342117 (0.061182) with: {'features__topic__lda__n_components': 16}
0.342351 (0.060884) with: {'features__topic__lda__n_components': 32}
__pipeline SGDRegressor:tune took 04:22:39

Best: 0.345709 using {'features__tfidf__vector__min_df': 1}
0.345709 (0.064010) with: {'features__tfidf__vector__min_df': 1}
0.343578 (0.062687) with: {'features__tfidf__vector__min_df': 10}
0.334686 (0.056161) with: {'features__tfidf__vector__min_df': 100}
__pipeline: SGDRegressor,tune took 01:53:11
```

### Validation

* SGD is the best performer, with R2 score of 0.37
* Evaluation uses R2 (scores below)
* 3-fold validation (save time and train on two-third majority)

```
SGD
Validation result: r2=0.3732233278447895 (median)
__validate took 00:42:44
__pipeline SGDRegressor:validate took 00:42:44

Ridge
Validation result: r2=0.2952067694919748 (median)
__validate took 00:42:50
__pipeline Ridge:validate took 00:42:50
```

### Test

* Evaluation uses R2 (same as validation), and median absolute error (MAE)
* Validation result is close to test result but performance is poor. This suggests underfitting
* SGD is the best performer, with R2 score of 0.34 and MAE of 2.2
```
#####  SGDRegressor:test  #####
Training...
'text' ColumnExtractor shape=(28969,)
'text' ColumnExtractor shape=(28969,)
'author_' PrefixColumnExtractor shape=(28969, 1678)
'token_count' ColumnExtractor shape=(28969, 1)
'token_length_mean' ColumnExtractor shape=(28969, 1)
'char_count' ColumnExtractor shape=(28969, 1)
'digit_char_ratio' ColumnExtractor shape=(28969, 1)
'alpha_char_ratio' ColumnExtractor shape=(28969, 1)
'upper_char_ratio' ColumnExtractor shape=(28969, 1)
'space_char_ratio' ColumnExtractor shape=(28969, 1)
'punctuation_char_ratio' ColumnExtractor shape=(28969, 1)
idfs len=18455, saved 'tmp/idf.txt'
topic_to_term shape=(8, 18455), saved 'tmp/topic_term.txt'
features len=20149
coefs len=20149, saved 'tmp/coefs.txt'
intercept=[0.09787454]
__train took 00:15:24
Testing...
'text' ColumnExtractor shape=(3219,)
'text' ColumnExtractor shape=(3219,)
'author_' PrefixColumnExtractor shape=(3219, 1678)
'token_count' ColumnExtractor shape=(3219, 1)
'token_length_mean' ColumnExtractor shape=(3219, 1)
'char_count' ColumnExtractor shape=(3219, 1)
'digit_char_ratio' ColumnExtractor shape=(3219, 1)
'alpha_char_ratio' ColumnExtractor shape=(3219, 1)
'upper_char_ratio' ColumnExtractor shape=(3219, 1)
'space_char_ratio' ColumnExtractor shape=(3219, 1)
'punctuation_char_ratio' ColumnExtractor shape=(3219, 1)
Test result: r2=0.34676911251904907, mae=2.216866481477075
__pipeline SGDRegressor:test took 00:16:50

#####  Ridge:test  #####
Training...
'text' ColumnExtractor shape=(28969,)
'text' ColumnExtractor shape=(28969,)
'author_' PrefixColumnExtractor shape=(28969, 1678)
'token_count' ColumnExtractor shape=(28969, 1)
'token_length_mean' ColumnExtractor shape=(28969, 1)
'char_count' ColumnExtractor shape=(28969, 1)
'digit_char_ratio' ColumnExtractor shape=(28969, 1)
'alpha_char_ratio' ColumnExtractor shape=(28969, 1)
'upper_char_ratio' ColumnExtractor shape=(28969, 1)
'space_char_ratio' ColumnExtractor shape=(28969, 1)
'punctuation_char_ratio' ColumnExtractor shape=(28969, 1)
idfs len=18455, saved 'tmp/idf.txt'
topic_to_term shape=(8, 18455), saved 'tmp/topic_term.txt'
features len=20149
coefs len=20149, saved 'tmp/coefs.txt'
intercept=[2.9110611]
__train took 00:16:25
Testing...
'text' ColumnExtractor shape=(3219,)
'text' ColumnExtractor shape=(3219,)
'author_' PrefixColumnExtractor shape=(3219, 1678)
'token_count' ColumnExtractor shape=(3219, 1)
'token_length_mean' ColumnExtractor shape=(3219, 1)
'char_count' ColumnExtractor shape=(3219, 1)
'digit_char_ratio' ColumnExtractor shape=(3219, 1)
'alpha_char_ratio' ColumnExtractor shape=(3219, 1)
'upper_char_ratio' ColumnExtractor shape=(3219, 1)
'space_char_ratio' ColumnExtractor shape=(3219, 1)
'punctuation_char_ratio' ColumnExtractor shape=(3219, 1)
Test result: r2=0.2665669801449393, mae=2.8413241055893064
__pipeline Ridge:test took 00:17:59
```

### Implementation

* Uses [`sklearn.pipeline.Pipeline`](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) to organise steps and reproduce results. See [classify_post_comment_count.py](classify_post_comment_count.py)
* Combine text and non-text features with [`sklearn.pipeline.FeatureUnion`](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html). 
* Wrote custom feature extractors by extending scikit-learn API. This allows better integration of pandas dataframe with `sklearn.pipeline.FeatureUnion`. See [`sklearnpd`](sklearnpd/) package. 
* Ease of parameter tuning: automatically run pipeline for each combination of parameters provided.

## Deploy
### Deploy as virtual environment

Install dependencies

`(venv) $ pip install -r /path/to/requirements.txt`

Run main with the following options

`(venv) $ python python classify_post_comment_count.py [model] [task]`

Model options
* `sgd`
* `ridge`

Task options
* `train`
* `validate`
* `tune`
* `test`

example:

`(venv) $ python python classify_post_comment_count.py sgd train`

Run unit tests

`(venv) $ python -m unittest`

See test cases for [`etl`](etl/tests/) and [`stringx`](stringx/tests/) packages.

### Deploy with setuptools

Install custom dependencies from public repositories.

Example [setup.py](setup.py)

```python
install_requires=[
    'sgcharts.stringx',
    ...
],
dependency_links=[
    'git+https://github.com/seahrh/sgcharts.stringx.git@master#egg=sgcharts.stringx-2.0.0'
]
```

Then `pip install` to add the custom dependency to the virtual environment.

```
pip install --process-dependency-links git+https://github.com/seahrh/sgcharts.stringx.git
```

## Future Work

TODO if I had more time :)

* Right-truncate data because almost no comments were recorded from Feb to May 2018. During this time, FB comments replaced the native system.
* Perform inference in two stages:
  * Classification task: predict post with non-zero comments
  * Regression task: predict number of comments for given post
* Filtering: set a max threshold for document frequency, numeric stopwords like '10AM'
* Add features from post metadata: day-of-week published
* Feature selection: chi-square
* Parameter tuning: SGD loss function, learning rate
* Increase test coverage
* Move config from code to file
* Logging to file
