# -*- coding: utf-8 -*-
"""
@author: Dipika Baad
"""

###Packages Included
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from nltk.corpus import stopwords
import gensim

from sklearn.cluster import KMeans
#from sklearn import metrics
#from scipy.spatial.distance import cdist
import numpy as np

###Tokenizer and stop words initialization
tokenizer = RegexpTokenizer(r'\w+')
en_stop = set(stopwords.words('english'))

#Read Input File "papers.csv" for NIPS DataSet
df1 = pd.read_csv("C:\\nips-papers\\papers.csv")

#Cleaning the data
texts = []
# loop through document list
for i in df1['paper_text']:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens and words less than 2 letters and remove numbers
    stopped_tokens = [i for i in tokens if (not i in en_stop and not str(i).isdigit() and len(str(i)) > 2 )]
    
    
    # add tokens to list
    texts.append(stopped_tokens)
    
df1['Cleaned_PaperText'] = pd.Series(texts, index = df1.index)

#Transforming to TFIDF Matrix        
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df1.paper_text)

##### Inertia error method - Loop to get error when number of clusters is set from 1-30
cluster_range = range( 1, 30 )
cluster_errors = []
for num_clusters in cluster_range:
   clusters = KMeans( num_clusters )
   clusters.fit( X )
   cluster_errors.append( clusters.inertia_ )
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

##Plot the graph (Output)
plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
plt.savefig("C:\\plot")
