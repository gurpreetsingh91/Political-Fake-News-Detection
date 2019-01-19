# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 18:28:44 2018

@author: GPSINGH
"""
import pickle
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from collections  import defaultdict
from time import strftime
import logging
from time import time
from pattern.db  import Datasheet
from random import shuffle
import string
import nltk
from numpy import genfromtxt
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from pattern.en import sentiment
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix
import itertools
from sklearn.metrics import confusion_matrix
from pattern.en import modality
from sklearn.naive_bayes import BernoulliNB    
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc


#To Plot the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]),   range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


#To extract Punctuation features from each news 
class Punct_Stats(BaseEstimator, TransformerMixin):
    """Extract punctuation features from each document"""

    def fit(self, x, y=None):
        return self

    def transform(self, text_fields):
        punct_stats = []
        punctuations = list(string.punctuation)
        print(punctuations)
        additional_punc = ['``', '--', '\'\'']
        punctuations.extend(additional_punc)
        for field in text_fields:
            puncts = defaultdict(int)
            for ch in field:
                if ch in punctuations:
                    puncts[ch]+=1
            punct_stats.append(puncts)
        return punct_stats


#Initiator to extract the headline and body features 
class HeadlineBodyFeaturesExtractor(BaseEstimator, TransformerMixin):
    """Extracts the components of each input in the data: headline, body, and POS tags for each"""
    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        headlineCol = posts[0].values.tolist()
        bodyCol = posts[1].values.tolist()
        punctuation = string.punctuation
        features = np.recarray(shape=(len(headlineCol),), dtype=[('headline', object), ('article_body', object), ('headline_pos', object), ('body_pos', object)])
        
        for i in range(0,len(headlineCol)):
            headline = headlineCol[i]
            article = bodyCol[i]
            features['headline'][i] = headline
            features['article_body'][i] = article

            tok_headline = nltk.word_tokenize(headline)
            features['headline_pos'][i] = (' ').join([x[1] for x in nltk.pos_tag(tok_headline)])

            tok_article = nltk.word_tokenize(article)
            features['body_pos'][i] = (' ').join([x[1] for x in nltk.pos_tag(tok_article)])
        
        return features

train_path = "C:\\Users\\GPSINGH\Pictures\\NewsAudit\\Reduced\\training_data.csv"
train_data = pd.read_csv(train_path, sep=',',header=None )
train,test = train_test_split(train_data , test_size = 0.2)

print(train)
train_labels = train[3]

del train[3]
train_texts = train

 
test_labels = test[3]
del test[3]
test_texts = test 

pipeline = Pipeline([
    # Extract the subject & body
    ('HeadlineBodyFeatures', HeadlineBodyFeaturesExtractor()),

    # Use FeatureUnion to combine the features from subject and body
    ('union', FeatureUnion(
        transformer_list=[

            #Pipeline for pulling punctuation feature from articles
            #Using FeatureHasher for both headline and the body
            ('punct_stats_headline', Pipeline([
                ('selector', ItemSelector(key='headline')),
                ('stats', Punct_Stats()),  
                ('vect', FeatureHasher(10)),   
            ])),

            ('punct_stats_body', Pipeline([
                ('selector', ItemSelector(key='article_body')),
                ('stats', Punct_Stats()),  
                ('vect', FeatureHasher(10)),  
            ])),

        ],
    )),

    # Use an Bernoulli Naive Bayes classifier as the Baseline Model
    ('clf', BernoulliNB()),
])
          
#Fitting the pipline to the training text and labels            
pipeline.fit(train_texts,train_labels)
y_pred = pipeline.predict(test_texts)

#Plotting the classification report
print(classification_report(y_pred,test_labels))
           
#Accuracy score
print("Accuracy score")
print(accuracy_score(test_labels, y_pred, normalize=False))



#AUC score from the ROC curve
print("AUC score")
print(roc_auc_score(test_labels, y_pred))
fpr, tpr, threshold = roc_curve(test_labels, y_pred)
roc_auc = auc(fpr, tpr)


#Plotting the ROC curve
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


labels = ['Objective', 'Sensationalist'] 

#Call to plot the confusion Matrix            
cnf_matrix = confusion_matrix(test_labels, y_pred)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=labels,
                      title='Confusion matrix')
plt.show()
            
            