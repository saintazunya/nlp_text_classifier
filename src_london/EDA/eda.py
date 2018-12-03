from collections import Counter
from collections import defaultdict
import pandas as pd
import asyncio
import random
import time
import math
from scipy.sparse import *
from scipy.sparse.linalg import svds
from sklearn.linear_model import LinearRegression
def log_idf(num_docus,total_docus):
    return math.log10(total_docus/(num_docus+1))
text_data_label=pd.read_csv('../../data/processed_text_data.csv').dropna()
text_data_label.index=range(len(text_data_label))
# term to document index
def tf_idf_data_prepare_sync():
    term_counter = {}
    idfdict = defaultdict(list)
    def generator():
        for x in text_data_label.iterrows():
            yield x
    for idx,(review,_) in generator():
        if not isinstance(review,str):
            print(review,'error')
            continue
        temp_counter=Counter(review.split())
        for key in temp_counter:
            idfdict[key].append(idx)
        term_counter[idx]=temp_counter
    return term_counter,idfdict

async def tf_idf_data_prepare():
    term_counter = {}
    idfdict = defaultdict(list)
    async def generator():
        for x in text_data_label.iterrows():
            yield x
    async for idx,(review,_) in generator():
        if not isinstance(review,str):
            print(review, 'error')
            continue
        temp_counter=Counter(review.split())
        for key in temp_counter:
            idfdict[key].append(idx)
        term_counter[idx]=temp_counter

    return term_counter,idfdict



loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
t=time.time()
term_counter,idfdict = loop.run_until_complete(tf_idf_data_prepare())
print(time.time()-t)

total_num_documents=len(text_data_label)
# this matrix is sparse,we store it in ijv format to save memory
rowidx=[]
coldix=[]
values=[]
# text_to_int
text_to_int=dict(zip(idfdict.keys(),range(len(idfdict))))
print(total_num_documents)
for idx in range(total_num_documents):
    length_of_document = sum(term_counter[idx].values())
    for key,value in term_counter[idx].items():
        # key is the term in one document, key is the frequency of that term in that document
        tf_idf_value=(value/length_of_document)*log_idf(len(idfdict[key]),total_num_documents)
        rowidx.append(text_to_int[key])
        coldix.append(idx)
        values.append(tf_idf_value)

tf_idf=csc_matrix(coo_matrix((values, (rowidx,coldix))))
# we assume term and document are composed from same latent vectors.
# do svd on the tf-idf matrix

(u,s,vt)=svds(tf_idf,k=128)
# u is the latent vector of terms dim: len of terms, k
# v is the latent vector of documents len of documents, k
# we want to classify the documents, thus we can use latent vector of documents as features to do regression.
# lets build a simple classify based on these features


reg=LinearRegression().fit(vt.transpose(), text_data_label.rating)
p=reg.predict(vt.transpose())
mse=sum((p-text_data_label.rating.values)**2)/len(p)


print(mse)


pass
