
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


![image](https://user-images.githubusercontent.com/92213377/215080648-107af007-905d-4d45-9a56-2a3a30f54709.png)
![image](https://user-images.githubusercontent.com/92213377/215080724-75967e06-0241-4955-a85e-a2f795821eb3.png)






## NLTK
The Natural Language Toolkit (NLTK) is a platform used for building Python programs that work with human language data for applying in statistical natural language processing (NLP). It contains text processing libraries for tokenization, parsing, classification, stemming, tagging and semantic reasoning.

## TENSORFLOW
TensorFlow is a Python-friendly open source library for numerical computation that makes machine learning and developing neural networks faster and easier.
What is TensorFlow used for?
The TensorFlow platform helps you implement best practices for data automation, model tracking, performance monitoring, and model retraining. Using production-level tools to automate and track model training over the lifetime of a product, service, or business process is critical to success.

## KERAS
Keras is a neural network Application Programming Interface (API) for Python that is tightly integrated with TensorFlow, which is used to build machine learning models. Keras' models offer a simple, user-friendly way to define a neural network, which will then be built for you by TensorFlow.

## BEAUTIFUL SOUP
Beautiful Soup is a python package and as the name suggests, parses the unwanted data and helps to organize and format the messy web data by fixing bad HTML and present to us in an easily-traversible XML structures. In short, Beautiful Soup is a python package which allows us to pull data out of HTML and XML documents.







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

beautiful soup

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

![image](https://user-images.githubusercontent.com/92213377/215081384-ca06223e-ec89-4866-a029-7047a652790a.png)

## Acknowledgements

  [https://www.kaggle.com/code/vaishnavi28krishna/twitter-analysis-using-dt-and-rfdtc](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)



## Appendix

A very crucial project in the realm of data science. Multiple concepts were used.


## Running Tests

To run tests, run the following command

```bash
  npm run test
```

We got an accuracy of 97 % in one model,and 89.5% in the subsequent one.We ran various epochs and used efficiet data cleansing techniques to get to this.

## Used By

The project is used by a lot of social media companies to analyse their market.



## Output

![image](https://user-images.githubusercontent.com/92213377/215080848-3f1daf05-4479-4f09-9b2d-f02c53dd0ff6.png)
![image](https://user-images.githubusercontent.com/92213377/215081130-50ac6984-176b-495a-bede-498bc50e302e.png)
![image](https://user-images.githubusercontent.com/92213377/215080955-86f19991-70ca-4b29-8260-057f8682e78d.png)


![image](https://user-images.githubusercontent.com/92213377/215081240-5b843c6b-8cde-45cb-9e45-c3980c127a2c.png)
![image](https://user-images.githubusercontent.com/92213377/215081291-90480ef2-280d-4b00-9380-95efb8b70754.png)
![image](https://user-images.githubusercontent.com/92213377/215081444-4919f6d0-ab05-40b8-bcc3-934d8ac83956.png)





