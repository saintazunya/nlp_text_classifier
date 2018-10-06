import pandas as pd
import numpy as np
import csv
from functools import reduce
from collections import defaultdict
freq=pd.read_csv('freqtext',delimiter='\t',header=None)
freq=freq.values.reshape((-1)).tolist()
f=lambda x: x[x.find('.')+1:].strip()
freq=[f(x) for x in freq]
file='tripadvisor.csv'
commalst=[',','.','?',':','"',"'",'!','@','#','%','*','+','-']
replace_comma=lambda x,y:x.replace(y,'')

changedict={str(x):str(x) for x in range(1,6)}
changedict=defaultdict(lambda : None,changedict)
changedictmapping=lambda x : changedict[x[0]] if isinstance(x,str) else None
with open(file,'r') as csvfile, open('processed_text_data.csv','w') as outfile:
    reader = csv.reader(csvfile)
    head=next(reader)
    writer=csv.writer(outfile)
    writer.writerow([head[8],head[12]])
    for row in reader:
        if not row[8] or not row[12]:
            continue
        temp1=reduce(replace_comma,[row[8]]+commalst).lower().split()
        temp1=[x for x in temp1 if x not in freq]
        temp1=' '.join(temp1)
        temp2=changedictmapping(row[12])
        writer.writerow([temp1,temp2])
        pass
pass