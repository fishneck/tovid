import numpy as np
import pandas as pd
import os
import json,time
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
from sklearn.decomposition import LatentDirichletAllocation as LDA
import pyLDAvis 
import pyLDAvis.sklearn 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


number_topics = 6
number_words = 20

data_dir = '../data/'
data = pd.read_csv(data_dir+'litcovid_words.csv')

save_plot=False

def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

def prune_topN_topic(N,model,dictionary):
    tokens = set()
    counts = model.components_.sum(axis=1)[:, np.newaxis]
    #print(counts)
    topic_short = []
    for topic_idx, topic in enumerate(model.components_):
        current_token = [dictionary[i] for i in topic.argsort()[:-N - 1:-1]]
        #print(current_token)
        tokens = tokens.union(set(current_token))
        value = sorted(topic/counts[topic_idx],reverse=True)[:100]
        pruned = dict(zip(current_token, value))
        topic_short.append(pruned)   
        #print(pruned)
    return topic_short  

def topic2cluster(topic_ids, cluster_map):
    #transform a list of topics by month to the cluster results
    clusters = []
    for i in range(len(topic_ids)):
        #print(topic_ids[i],cluster_map[6*i:6*(i+1)],6*i,6*(i+1))
        cluster_by_month = {}
        for j in set(cluster_map[6*i:6*(i+1)]):
            topic2cluster_id = [i_ for i_,val in enumerate(cluster_map[6*i:6*(i+1)]) if val==j]
            cluster_cnt = [np.sum(topic_ids[i]==obj) for obj in topic2cluster_id]
            total_cnt = np.sum(cluster_cnt)
            cluster_by_month[j] = total_cnt/topic_ids[i].shape[0]
            #print(j,topic2cluster_id,cluster_cnt,total_cnt)
        clusters.append(cluster_by_month)
    return clusters

##############################################      
#### select number of topics using titles ####
##############################################    

stopwords=[]
count_vectorizer = CountVectorizer(ngram_range=(1, 1),stop_words='english')
count_data = count_vectorizer.fit_transform(data['title'])

words = count_vectorizer.get_feature_names()
total_counts = np.zeros(len(words))
for t in count_data:
    total_counts+=t.toarray()[0]
count_dict = (zip(words, total_counts))
count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:100]
print(count_dict)

built_in = count_vectorizer.get_stop_words()
stopwords={count_dict[i][0] for i in range(25)}


count_vectorizer = CountVectorizer(stop_words=set(built_in).union(stopwords))
count_data = count_vectorizer.fit_transform(data['title'])
print(len(count_vectorizer.vocabulary_))
   
lda= LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)
print("Topics found via LDA:")
print_topics(lda, count_vectorizer, number_words)


model = pyLDAvis.sklearn.prepare(lda,count_data,count_vectorizer)
pyLDAvis.save_html(model, '../model/title_lda.html')


min_p = 11111.
p=[]
start = 5
end = 100
for i in range(start,end):
    print(i)
    number_topics=i
    lda_ = LDA(n_components=number_topics, n_jobs=-1)
    lda_.fit(count_data)
    curr_p=lda_.perplexity(count_data)
    p.append(curr_p)
    if curr_p<min_p:
        best_lda = lda_

plt.plot(list(range(start,end)),p)
plt.ylabel('perplexity')
plt.xlabel('number of topics in LDA')

if save_plot:
    plt.plot(list(range(start,end)),p)
    plt.savefig("img/perplexity.jpeg")
    

####################################################       
#### monthly topic modeling with LDA and abstrac####
####################################################


data = data[data.abstract.map(lambda x: len(str(x)))>50]
data = data.sort_values(by='month')
data = data[pd.notnull(data.abstract)]


lda_by_month = []
dict_by_month = []
topic_allocation = []
for i in range(1,13):
    print('--------' + str(i)+'----------------')
    count_data = count_vectorizer.fit_transform(data[data.month==i]['abstract'])
    dict_by_month.append(count_vectorizer.get_feature_names())
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(count_data)
    topic_allocation.append(lda.transform(count_data).argmax(axis=1))
    lda_by_month.append(lda)
    #print("Topics found via LDA:")
    #print_filter_topics(lda, count_vectorizer, number_words,filters)
    
    
N = 100
topic_representation = prune_topN_topic(N,lda_by_month[0],dict_by_month[0])

for i in range(1,len(lda_by_month)):
    topic_representation.extend(prune_topN_topic(N,lda_by_month[i],dict_by_month[i]))

topic_df = pd.DataFrame(topic_representation)

drop_fea = ['cases','virus',
        'high', 'time', 'based', 'patient', 'viral', 'new','use','syndrome',
        'including','associated', 'using','methods','95', 'ci','la','en','et','des','les','related']

color = np.arange(1,13).repeat(6)
pca = PCA(n_components=2)
pca.fit(topic_df.drop(drop_fea,axis=1).fillna(0))
X_embed = pca.transform(topic_df.drop(drop_fea,axis=1).fillna(0))

sc = plt.scatter(X_embed[:,0],X_embed[:,1],c=color,cmap='plasma')
plt.colorbar(sc)
plt.xlabel('PCA-1')
plt.ylabel('PCA-2')
plt.title('top '+  str(N)+ ' topic words decomposition')

if save_plot:
    plt.plot(list(range(start,end)),p)
    plt.savefig("img/PCA_"+str(N)+".jpeg")
    

cluster = 7
model = KMeans(n_clusters=cluster)
model.fit(topic_df.drop(drop_fea,axis=1).fillna(0))

    
# Prediction on the entire data
all_predictions = model.predict(topic_df.drop(drop_fea,axis=1).fillna(0))

#check the cluster result
for i in range(cluster):
    print(topic_df.drop(drop_fea,axis=1)[all_predictions==i].sum().sort_values(ascending=False)[:15].index)

proportion = topic2cluster(topic_allocation,all_predictions)



#plot the proportion
cmap = plt.get_cmap('jet')
y = [item[0]  if 0 in item.keys() else 0. for item in proportion]

plt.bar(np.arange(1,13),
        y,
        color=cmap(0),
        label= ' | '.join(topic_df.drop(drop_fea,axis=1)[all_predictions==0].sum().sort_values(ascending=False)[:5].index)
        )

last_y = y
for i in range(1,cluster):
    y = [item[i]  if i in item.keys() else 0. for item in proportion]
    topic_words = ' | '.join(topic_df.drop(drop_fea,axis=1)[all_predictions==i].sum().sort_values(ascending=False)[:5].index)
    plt.bar(np.arange(1,13),
        y,
        color=cmap(i/cluster),
        bottom=last_y,
        label=topic_words
        )
    last_y = [k + j for k, j in zip(y, last_y)]


plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)    
plt.xlabel('month')
plt.ylabel('proportion')
plt.title('topic evolution in 2020')

plt.savefig("img/proprtion.jpg")


