# -*- coding: utf-8 -*-
"""
@author: Dipika Baad
"""

###Packages Included
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from nltk.corpus import stopwords
import gensim

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

#Create corpus of words
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
#Storing the corpus in one of the column
df1['Corpus'] = pd.Series(corpus, index = df1.index)
#Train the model for 20 topics
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, id2word = dictionary, passes=20)
## If you want to check the topic words
#print(ldamodel.print_topics(num_topics=2, num_words=4))

#save model
ldamodel.save('C:\\model.atmodel')
#Load model
model = gensim.models.ldamodel.LdaModel.load('C:\\model.atmodel')

######Creating the empty columns to hold the top words and top topics
Top_words = []
topic_words = []
top_topic = []
for index,row in df9.iterrows():
    topic_words.append('')
    top_topic.append('')
df1['Top_Words1'] = pd.Series(topic_words, index = df1.index)
df1['Top_Topic1'] = pd.Series(top_topic, index = df1.index)

for index,row in df1.iterrows():
    words = [] # to store the list of words for this given topic
    #max1 array to hold the topic numbers for this paper
    max1 = []

    #Get the document topics by model
    t = model.get_document_topics(row['Corpus'])
    #t had list of tuple(topic number, probability)
    #Loop through the topics found above
    for k in t:
        #Show topic gives the list of tuples (word, probabibility in topic)
        t1 = model.show_topic(k[0])
        max1.append(k[0])
        for i in t1:
            words.append(i[0])
    #Set the top words for the paper
    df1.set_value(index,"Top_Words1", words)
    #Set the top topics for the paper
    df1.set_value(index,"Top_Topic1", max1)

#List of top words for each row converted into sentence of words seperated by space
for index,row in df1.iterrows():
    m= []
    for i in row["Top_Words1"]:
        m.append(str(i))
        str1 = (" ").join(m)
        df1.set_value(index,"Top_Words1", str1)
        
#Output
#Store the output as csv with sep="\t"
f1 = open("C:\\paper_20topicWords_new1.csv","w")
pd.DataFrame(df1).to_csv(f1,columns=["id","paper_text","Top_Words1"],sep="\t")
f1.close()
