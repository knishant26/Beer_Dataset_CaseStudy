# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 11:38:18 2020

@author: 91956
"""
import datetime
datetime.datetime.now()

import pandas as pd
import numpy as np
import seaborn as sns

import nltk, string
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score

beer_df_original= pd.read_csv('train.csv')
# beer_df_original.columns

beer_df= beer_df_original[['review/overall', 'review/text']]
beer_df.dropna(inplace= True)
# beer_df['review/text'].isnull().sum()


def text_process(text):
    nopunc= [char for char in text if char not in string.punctuation]
    nopunc= ''.join(nopunc)
    return ' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])

beer_df['Filtered Review Text'] = text_process(beer_df['review/text'])

print('processing done')

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

beer_df['Filtered Review Text'] = beer_df['Filtered Review Text'].apply(lemmatize_text)

print('lemmatization done')

cvec = CountVectorizer(min_df=.005, max_df=.9, ngram_range=(1,2))
cvec.fit(beer_df['Filtered Review Text'])

print(len(cvec.vocabulary_))


cvec_counts = cvec.transform(beer_df['Filtered Review Text'])

transformer = TfidfTransformer()

transformed_weights = transformer.fit_transform(cvec_counts)

transformed_weights = transformed_weights.toarray()
vocab = cvec.get_feature_names()

model = pd.DataFrame(transformed_weights, columns=vocab)
model['Keyword'] = model.idxmax(axis=1)
model['Max'] = model.max(axis=1)

model = pd.merge(beer_df, model, left_index=True, right_index=True)

#max occuring words
occ = np.asarray(cvec_counts.sum(axis=0)).ravel().tolist()
counts_df = pd.DataFrame({'Term': cvec.get_feature_names(), 'Occurrences': occ})
counts_df.sort_values(by='Occurrences', ascending=False).head(25)


ml_model = model.drop(['review/text', 'Filtered Review Text', 'Keyword', 'Max', 'Sum'], axis=1)

X = ml_model.drop('review/overall', axis=1)
y = ml_model['review/overall'].astype(str)

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size= 0.3)


sgd= SGDClassifier()
sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))

















