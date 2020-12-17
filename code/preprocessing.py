import numpy as np
import pandas as pd
import os
import json,time


data_dir = '../data/'

def process_json(json):
    #abstract, doi, title journal
    index = json['_id']
    journal = json['journal']
    
    try: 
        doi = json['passages'][0]['infons']['journal']    
    except:
        doi=''
        
    try:
        title = json['passages'][0]['text']
    except:
        title = '' 
        
    abstract = ''    
    for item in json['passages']:
        if 'type' in item['infons'].keys() and item['infons']['type']=='abstract':
            abstract = item['text']
    
    if len(abstract)<10 and len(json['passages'])>1 and len(json['passages'][1]['text'])>80:
        abstract = json['passages'][1]['text']
        
    date = float(json['created']['$date'])/1000
    date = time.strftime('%Y-%m-%d', time.gmtime(timestamp))
    
    
    return {'index':index, 'title':title, 'abstract':abstract, 'doi':doi, 'journal':journal,'date':date}




def doi_date(doi):
    month = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,\
             'May':5,'Jun':6,'Jul':7,'Aug':8,\
             'Sep':9,'Sept':9,'Oct':10,'Nov':11,'Dec':12}
    journal = re.search('Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec',doi)
    m=-1
    if journal:
        m = journal.group(0)
        m = month[m]
    else:
        doi = doi.replace('.','')
        date = doi.split(';')[1]
        #print(doi,date)
        try:
            date = date.replace('2020','')
            m = int(date.split(' ')[2])
            #print(m,date)
        except:
            '''
            if len(doi.split(';'))>2:
                date = doi.split(';')[2]
                m = int(date.split(' ')[1])
                #print(date)
            '''
            m=-1
    return m



if __name__ == "__main__": 
    with open(data_dir+'litcovid2BioCJSON.json') as json_file:
        json_file = json.load(json_file)
    print(len(json_file[1]))
    for i in range(1,len(json_file[1])):
        #len(json_file[1])
        d = process_json(json_file[1][i])
        row = pd.DataFrame([d.values()],columns=d.keys())
        data = data.append(row,ignore_index=True)
        if i%1000 ==0:
            print(i)
    data = data[(data.doi!='')&(data.title!='')&(data.abstract!='')]
    data['month'] = data.doi.map(doi_date)
    data = data.drop('date',axis=1)
    data = data[(data.month>0) &(data.month<13)]
    data.to_csv(data_dir+'litcovid_words.csv',index=False)
