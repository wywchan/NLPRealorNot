# Real or Not? NLP with Disaster Tweets
by Wayne Chan and Kavan Pandya

## Introduction

The Real or Not tweet challenge is an ongoing competition hosted by Kaggle. This challenge is designed to be an introductory problem for those getting started with NLP using real Twitter data.

We approached this problem using common Scikit-Learn classifiers and then comparing the best performing ones with a neural network. The challenge scoring is determined by F1 score.

Ultimately, our best performing solution was a Multinomial Naive Bayes model which had an F1 score of 0.79 representing a reasonable amount of recall as well as precision.

### Methodology

Our approach was to clean the data by first removing any symbols, urls and numbers.  We then imputed missing values in the keyword and location columns by comparing the body text with a list of existing keywords and locations. The keyword, location and tweet text were then combined to create the corpus for our modelling.

We established a baseline result using selected classifiers from the Scikit-Learn library and performed a base parameter grid search to determine which classifier and vectorizer method worked best. After choosing the top performers from this group, we further tuned them to get improved results. We then created a basic neural network and compared it against the best performing classifier from the grid search and made predictions to submit.

Some challenges we came across included F1 score not being among the selectable metrics in Tensorflow as well as difficulties using a GloVe vectorizer. The neural network was originally setup to use the GloVe vectorizer but initial results were not promising. We ultimately had to use the default vectorizer from Keras in order to get the network to work properly.

---

### Data Dictionary

|Feature|Type|Description|
|---|---|---|
|id|int64|Unique identifier for the tweet|
|keyword|object|Keyword related to the tweet|
|location|object|Location the tweet was sent from|
|text|object|Content of the tweet|
|target|int64|1 if tweet relates to a disaster, 0 if not|

#### Provided Data

Provided datasets from Kaggle:

- [Training dataset](./train.csv)
- [Test dataset](./test.csv)

---

### Conclusion

The Multinomial Naive-Bayes model was the best solution based on our results and achieved a F1 score of 0.79 on the test dataset which is a reasonably good result. While we did attempt a neural network solution as well it was vastly overfit. Despite our attempts to simplify it, the predictions ended up always outputting the same value. Either our data is too simple for a neural network or we need to work on further simplifications to the neural network config.

### Future Improvements

Areas of improvement in the future include better inputing of missing location and keywords, additional hyperparameter tuning on the Scikit-Learn classifiers and also additional tuning for the neural network. Furthermore, some tweets had hashtags which could be compared against keywords.

It wasn't necessary to visualize an ROC curve or identify which ngrams were most significant for the challenge but these could be included in the future for completeness.
