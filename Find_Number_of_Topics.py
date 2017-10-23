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

tokenizer = RegexpTokenizer(r'\w+')
en_stop = set(stopwords.words('english'))


df9 = pd.read_csv("C:\\papers.csv")

texts = []
# loop through document list
for i in df9['paper_text']:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if (not i in en_stop and not str(i).isdigit() and len(str(i)) > 2 )]
    
    # stem tokens
    #stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stopped_tokens)
    
df9['Cleaned_PaperText'] = pd.Series(texts, index = df9.index)
        
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df9.paper_text)
#km = KMeans(n_clusters=10, init='k-means++', max_iter=100)
#km.fit(X)
#km1.labels_

# k means determine k
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
 
### One more method for 

for n_cluster in range(2, 11):
    kmeans = KMeans(n_clusters=n_cluster).fit(X)
    label = kmeans.labels_
    sil_coeff = silhouette_score(X, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
##dictionary = corpora.Dictionary(texts)
#corpus = [dictionary.doc2bow(text) for text in texts]
#df9['Corpus'] = pd.Series(corpus, index = df9.index)

##### Inertia error method
cluster_range = range( 1, 30 )
cluster_errors = []
for num_clusters in cluster_range:
   clusters = KMeans( num_clusters )
   clusters.fit( X )
   cluster_errors.append( clusters.inertia_ )
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
