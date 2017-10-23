# -*- coding: utf-8 -*-
"""
@author: Dipika Baad
"""

###Packages Included
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from nltk.corpus import stopwords
import gensim

#Reading Input Files
#Read the papers.csv
df2 = pd.read_csv("C:\\nips-papers\\papers.csv")
#Read the visualization file created by other module where Level 1 - Year-Range, Level 3 is Paper Title (Only these fields are used)
df1 = pd.read_csv("C:\\visualizations.csv")

##Cleaning Data for Topic Modeling
tokenizer = RegexpTokenizer(r'\w+')
en_stop = set(stopwords.words('english'))



paper_text1 = []
ids=[]
for index, row in df1.iterrows():
    m = df2[df2["title"] == row["Level3"]]
    ids.append(m.id.iloc[0])
    paper_text1.append(m.paper_text.iloc[0])
df1['Paper_ID'] = pd.Series(ids, index = df1.index)
df1['Paper_text'] = pd.Series(paper_text1, index = df1.index)

corpus = []
tokenizer = RegexpTokenizer(r'\w+')
en_stop = set(stopwords.words('english'))
texts = []
    ################################### Remove numbers and single letter words
    # loop through document list
for i in df1['Paper_text']:
        
        # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    
        # remove stop words from tokens
    stopped_tokens = [i for i in tokens if (not i in en_stop and not str(i).isdigit() and len(str(i)) > 2 )]
        
        # add tokens to list
    texts.append(stopped_tokens)
    
df1['Cleaned_PaperText'] = pd.Series(texts, index = df1.index)

#Creating an empty column for holding the corpus data
corpus1 =[]
for index,row in df1.iterrows():
    corpus1.append('')
df1["Corpus1"] = pd.Series(corpus1, index = df1.index)

#Finding unique Year-Ranges
Unique_YearSubsets = df1.Level1.unique()

##Creating the empty column to store the kmeans cluster number
kvals = []
for index,row in df1.iterrows():
        kvals.append('')
df1["KCluster"] = pd.Series(kvals, index = df1.index)

#Loop through the Year-range to run kmean clustering on each subset
for Year_Subset in Unique_YearSubsets:
    df_Year_Subset = df1[df1["Level1"]==Year_Subset]
    ###FInding the subset dataframe with unique paper_ids
    Unique_PaperIDs_By_Year_Subset = df_Year_Subset.Paper_ID.unique()
    df_mean = pd.DataFrame()
    for i in Unique_PaperIDs_By_Year_Subset:
        
        df_temp = df1[df1["Paper_ID"] == i]
        df_temp1 = df_temp.head(1)
        T = [df_mean, df_temp1]
        df_mean = pd.concat(T)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df_mean.paper_text)
    ##Run Kmeans Clustering with n=10
    km = KMeans(n_clusters=10, init='k-means++')
    km1 = km.fit(X)
    km2 = km1.labels_
    i = 0
    ###Creating empty column for holding the cluster numbers for this subset
    kvals =[]
    for index,row in df_mean.iterrows():
        kvals.append('')
    df_mean["KCluster"] = pd.Series(kvals, index = df_mean.index)
    ##Setting the cluster number for df_mean
    for index,row in df_mean.iterrows():
        df_mean.set_value(index,"KCluster", km2[i])
        i+= 1
    ##Assigning to the actual paper_id in the main dataframe
    for index,row in df_mean.iterrows():
        for k in df1.loc[df1.Paper_ID == row["Paper_ID"], 'KCluster'].index:
            df1.set_value(k,"KCluster", row["KCluster"])
            
#Finding unique cluster IDs
Unique_Kmeans = df1.KCluster.unique()

###Loop to go through each cluster of each year-range to train the LDA model
for unique_cluster in Unique_Kmeans:
        
    for Year_Subset in Unique_YearSubsets:
        df_Year_Subset = df1[df1["Level1"]==Year_Subset]
        df_Year_Subset = df_Year_Subset[df_Year_Subset["KCluster"] == unique_cluster]
        Unique_PaperIDs_By_Year_Subset = df_Year_Subset.Paper_ID.unique()
        texts = []
        for i in Unique_PaperIDs_By_Year_Subset:
            
            df_temp = df1[df1["Paper_ID"] == i]
            df_temp1 = df_temp.head(1)
            texts.append(df_temp1["Cleaned_PaperText"].tolist()[0])
            
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        for i in Unique_PaperIDs_By_Year_Subset:
            j = 0
            for k in df1.loc[df1.Paper_ID == i, 'Corpus1'].index:
                df1.set_value(k,"Corpus1", corpus[j])
            j+=1
        try:
            #Run the topic modeling 
            ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=20)
        except:
            continue
        #Save the model with unique names bu Year_Subset (2002-2005 for eg) and with cluster number like (0,1,... or 9)
        ldamodel.save('C:\\kmeans\\model'+Year_Subset+'K'+unique_cluster+'.atmodel')
    

Top_words = []
topic_words = []
top_topic = []
for index,row in df1.iterrows():
    topic_words.append('')
    top_topic.append('')
df1['Top_Words1'] = pd.Series(topic_words, index = df1.index)
df1['Top_Topic1'] = pd.Series(top_topic, index = df1.index)

for index,row in df1.iterrows():
    words = []
    Year_Range = row["Level1"]
    kmeans = row["KCluster"]
    try:
        model = gensim.models.ldamodel.LdaModel.load('C:\\Dipika\\Eindhoven\\Quarter1\\Web Information Retrieval\\Web IR Project\\kmeans\\model'+Year_Range+'K'+str(kmeans)+'.atmodel')
    except:
        continue
    t = model.get_document_topics(row['Corpus1'],minimum_probability=0.2)

    max1 = []
    for k in t:
        
        t1 = model.show_topic(k[0])
        max1.append(k[0])
        cnt = 0
        #If jus one topic then take all the top 10 words
        if len(t) == 1:
            max_cnt = 10
        else:
        #Else take top 5 words from each topic found for the paper
            max_cnt = 5
        for i in t1:
            words.append(i[0])
            cnt+=1
            if cnt == max_cnt:
                break
    #Top_words.append(words)
    df1.set_value(index,"Top_Words1", words)
    df1.set_value(index,"Top_Topic1", max1)
    
f2 = open("C:\\viz_with_clusters.csv","w")
pd.DataFrame(df1).to_csv( f2,columns = ["Level1","Level2","Level3","Level4","Federal","Total","Paper_ID","Top_Words1","Top_Topic1","KCluster"], index=False)
f2.close()
