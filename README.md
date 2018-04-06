# POCOJO

POsts-COmments-JObs dataset

## Deploy as virtual environment

Install dependencies

`(venv) $ pip install -r /path/to/requirements.txt`

Run unit tests

`(venv) $ python -m unittest`

[Test cases](etl/tests/)

## Data

Size
* 7,074 posts (worked around the page limit in the posts api)
* Downloaded comments for each post
* 4,376 job postings

Comment-count by quartile
* 25% of posts have zero comments
* 50% of posts have 2 or less comments
* 75% of posts have 5 or less comments

## Task 1

Classify posts by the number of comments. Get insights into topics or writing style that drives comments.

### Classes

Comment count by range interval (based on quartile analysis)
* `lo` = [0,2]
* `mi` = [3,5]
* `hi` = 6 or more

### Splitting

* Test set holdout at 10% of posts (~700)
* Reproducible random split, because random seed is fixed

### Preprocessing

* Parse json to extract fields
* Remove HTML tags
* Convert HTML entities to ASCII
* tf-idf weights
    * Lowercase
    * Remove stopwords
    * Remove punctuation
    * Stemming (Porter)
* Word count
* Average word length
* Author name (one-hot encoded)
* Document frequency thresholding
* Count comments by post

Sketch of how features are combined
![](docs/features.svg?raw=true)

### Classifiers

* Multinomial Naive Bayes
* Linear SVM
* Random forest
* Gradient boosting machine

### Parameter tuning

* Grid search
* Evaluation uses macro F1 (scores below)
* Due to time constraint, only tuned one variable: document frequency thresholding (best=.01)

```
#####  MultinomialNB  #####

Grid search

Best: 0.410961 using {'features__tfidf__vector__min_df': 0.01}

0.229051 (0.000118) with: {'features__tfidf__vector__min_df': 1}

0.410961 (0.008981) with: {'features__tfidf__vector__min_df': 0.01}

0.400982 (0.007489) with: {'features__tfidf__vector__min_df': 0.05}

Time taken 00:29:49

#####  LinearSVC  #####

Grid search

Best: 0.264455 using {'features__tfidf__vector__min_df': 0.01}

0.253554 (0.034195) with: {'features__tfidf__vector__min_df': 1}

0.264455 (0.049938) with: {'features__tfidf__vector__min_df': 0.01}

0.219819 (0.022228) with: {'features__tfidf__vector__min_df': 0.05}

Time taken 00:31:12

#####  RandomForestClassifier  #####

Grid search

Best: 0.390547 using {'features__tfidf__vector__min_df': 0.01}

0.374912 (0.016261) with: {'features__tfidf__vector__min_df': 1}

0.390547 (0.012971) with: {'features__tfidf__vector__min_df': 0.01}

0.385066 (0.015266) with: {'features__tfidf__vector__min_df': 0.05}

Time taken 00:30:00

#####  GradientBoostingClassifier  #####

Grid search

Best: 0.424758 using {'features__tfidf__vector__min_df': 1}

0.424758 (0.005731) with: {'features__tfidf__vector__min_df': 1}

0.419609 (0.013092) with: {'features__tfidf__vector__min_df': 0.01}

0.417290 (0.006590) with: {'features__tfidf__vector__min_df': 0.05}

Time taken 00:46:34
```

### Validation

* Evaluation uses macro F1. As macro F1 gives equal weight to all classes, it is stricter than micro F1
* 3-fold validation (save time and train on two-third majority)

```

#####  MultinomialNB  #####

Validation result

f1_macro=0.4084200536715857 (median)

#####  LinearSVC  #####

Validation result

f1_macro=0.23223945151462536 (median)

#####  RandomForestClassifier  #####

Validation result

f1_macro=0.39869317447985214 (median)

#####  GradientBoostingClassifier  #####

Validation result

f1_macro=0.4257275226348023 (median)

```

### Test

* Evaluation uses macro F1 (same as validation)
* Validation result is close to test result but performance is poor. This suggests underfitting
* Class imbalance: the largest class `lo` is best performing
* Worst model is linear SVM. This suggests decision boundaries are non-linear
* Best model is GBM, but its pipeline also takes the longest time
* Performance of NB is close to GBM, but almost twice as fast

```

#####  MultinomialNB  #####

Test result

**f1_macro=0.4059017962638907**

f1_micro=0.5466101694915254

             precision    recall  f1-score   support

         hi       0.41      0.42      0.42       169

         lo       0.61      0.84      0.71       363

         mi       0.32      0.06      0.10       176

avg / total       0.49      0.55      0.49       708

Time taken 00:07:31

#####  LinearSVC  #####

Test result

**f1_macro=0.264795693706809**

f1_micro=0.3531073446327684

             precision    recall  f1-score   support

         hi       0.00      0.00      0.00       169

         lo       0.73      0.26      0.38       363

         mi       0.27      0.89      0.41       176

avg / total       0.44      0.35      0.30       708

Time taken 00:08:09

#####  RandomForestClassifier  #####

Test result

**f1_macro=0.377749821261618**

f1_micro=0.5098870056497176

             precision    recall  f1-score   support

         hi       0.39      0.31      0.34       169

         lo       0.57      0.81      0.67       363

         mi       0.25      0.08      0.12       176

avg / total       0.45      0.51      0.46       708

Time taken 00:07:32

#####  GradientBoostingClassifier  #####

Test result

**f1_macro=0.4324739554379346**

f1_micro=0.5720338983050848

             precision    recall  f1-score   support

         hi       0.50      0.39      0.44       169

         lo       0.60      0.89      0.72       363

         mi       0.38      0.09      0.14       176

avg / total       0.52      0.57      0.51       708

Time taken 00:12:33

```

### Implementation

* Uses [`sklearn.pipeline.Pipeline`](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) to organise steps and reproduce results
* Combine text and non-text features with [`sklearn.pipeline.FeatureUnion`](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html)
* Ease of parameter tuning - automatically run pipeline for each combination of parameters provided   
* see [classify_post_comment_count.py](classify_post_comment_count.py)

In the script, toggle `__is_tuning = True` or `__is_tuning = False`. Turn tuning off to run the validate-and-test pipeline instead.
```
(venv) $ python classify_post_comment_count.py
```
## Future Work

TODO if I had more time :)

* Add features from post metadata e.g. day-of-week published
* Feature selection e.g. chi-square
* Parameter tuning
* Hand pick stopwords
* Show most important features per class
* Topic modelling on selected features - see topics that drive comments
* Increase test coverage
* Move config from code to file
* Logging to file

## TODO Task 2

Given a post, find similar jobs. This is required for job recommendation:
* Document vectors are used for content-based recommendation.
* Similarity score is used for item-based collaborative filtering.
