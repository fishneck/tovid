from gensim.corpora import Dictionary
from gensim.models.ldaseqmodel import LdaSeqModel
import re
import numpy as np
import pandas as pd
import os
import json,math
import matplotlib.pyplot as plt
from gensim.parsing.preprocessing import STOPWORDS


dateiname1_dtm = '../model/abstract_1gram.model'
data_dir = '../data/'

data = pd.read_csv(data_dir+'litcovid_words.csv')
data = data.sort_values(by='month')
data = data[data.abstract.map(lambda x: len(str(x)))>50]

corpus_ = data.abstract.map(lambda x: re.sub('\(|\)|:|\,|\?|\.|\!','',str(x).lower()).split(' '))

stopwords = {'19','2019','20','2020',\
 'acute','analysis','care','case','clinical',\
 'coronavirus','cov','covid','disease','health',\
 'impact','infection','management','novel',\
 'outbreak','pandemic','patients','review','risk',\
 'sars','severe','study','treatment',\
 '2019-ncov','covid-19','coronavirus',\
 'cases', 'respiratory', 'results', 'data', 'virus','room',\
 'high', 'time', 'based', 'patient', 'new','use','syndrome',\
 'including','associated', 'using','methods','95', 'ci',\
 'la','en','et','des','les','el','le','drugs','related'}
my_stop_words = STOPWORDS.union(set(stopwords))
my_stop_words


dictionary = Dictionary(corpus_)
# remove unwanted word ids from the dictionary in place
dictionary.filter_extremes(no_below=50, no_above=0.9)
del_ids = [k for k,v in dictionary.items() if v in my_stop_words]
dictionary.filter_tokens(bad_ids=del_ids)


corpus = [dictionary.doc2bow(s) for s in corpus_]

print('Number of unique tokens: '+ str(len(corpus)))
print('Number of documents: '+str(len(dictionary)))

timeslice = list(data.month.value_counts(sort=False))
number_topics = math.floor(len(corpus)/len(dictionary))

ldaseq = LdaSeqModel(corpus=corpus,\
                     id2word=dictionary,\
                     time_slice=timeslice,\
                     num_topics=number_topics)

# it takes about 10 hours to train the model
ldaseq.save(dateiname1_dtm)
