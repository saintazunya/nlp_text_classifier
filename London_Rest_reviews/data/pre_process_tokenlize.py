from nltk.tokenize import sent_tokenize, word_tokenize

import pandas as pd
import numpy as np
import csv
from functools import reduce
from collections import defaultdict
import pickle
def loadGlovetoken(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile,'r')
    model = set()
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        #embedding = np.array([float(val) for val in splitLine[1:]])
        model.add(word)
    print( "Done.",len(model)," words loaded!")
    return model

file='tripadvisor.csv'

changedict={str(x):str(x) for x in range(1,6)}
changedict=defaultdict(lambda : None,changedict)
changedictmapping=lambda x : changedict[x[0]] if isinstance(x,str) else None
# tokenlize and only maintain the vords in Glove.
model=loadGlovetoken('/home/zcf/Documents/Graduate/NLP/glove.twitter.27B.200d.txt')
pickle.dump(model,open('tokenlst.pkl','wb'))
with open(file,'r') as csvfile, open('processed_text_data.csv','w') as outfile:
    reader = csv.reader(csvfile)
    head=next(reader)
    writer=csv.writer(outfile)
    writer.writerow([head[8],head[12]])
    for row in reader:
        if not row[8] or not row[12]:
            continue
        temp1=' '.join([x for x in word_tokenize(row[8].replace('\\','')) if x in model])
        temp2 = changedictmapping(row[12])
        writer.writerow([temp1,temp2])
        pass