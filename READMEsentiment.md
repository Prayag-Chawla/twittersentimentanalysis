
# Twitter Sentiment Analysis

In this project, I have done two ways, one using LSTM, and the other one using DT and RFDTC. I have attempted to predict the sentiment of a twitter word byv yraining and testing our model on the dataset, and I have also speified the importance of a specified word. There are two approaches, in this readme, I will generally be briefing about all the concepts used and the dictioranies and the combined outpyt of both thr notebook.
We have used variois LST layers like DENSE, EMBEDDING, etc
## Description

- Advanced data science analysis was used in this project. variousdictionaries and high concepts were also used.

- various graphical approach to visualise data after the step of data cleansing and data reading.
- Training and testing data, and checking the accuracy
- Usage of various LSTM layers and random forest classifier algorithm
- Usage of various data cleansing techniques, use of concept of padding.
- Usage of various aactivation functions like softmax, relu, etc.


## Usage and Installation

```import numpy as np
import numpy as np 
import pandas as pd 
import os

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import word_tokenize

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, SpatialDropout1D, Embedding, LSTM

lemmatizer = WordNetLemmatizer()

import matplotlib.pyplot as plt
import seaborn as sns
from nltk import FreqDist

from wordcloud import WordCloud

from nltk import FreqDist

from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=50, max_features="auto")
rf_model.fit(X_train, y_train)

from sklearn.metrics import roc_auc_score
```


## Acknowledgements

  [https://www.kaggle.com/code/vaishnavi28krishna/twitter-analysis-using-dt-and-rfdtc](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)



## Appendix

A very crucial project in the realm of data science. Multiple concepts were used.


## Running Tests

To run tests, run the following command

```bash
  npm run test
```

We got an accuracy of 97 % in this model. We ran various epochs and used efficiet data cleansing techniques to get to this.

## Used By

The project is used by a lot of social media companies to analyse their market.


