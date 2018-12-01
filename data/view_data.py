#spit data into train and eval
from collections import defaultdict
import pandas as pd
from sklearn.utils import shuffle
text_data_label=pd.read_csv('processed_text_data.csv')
print(text_data_label.shape)
#datalen=len(text_data_label)
data=text_data_label
'''
text_data_label=data[['review_text','rating']]
changedict={str(x):str(x) for x in range(1,6)}
changedict=defaultdict(lambda : None,changedict)
text_data_label['rating']=text_data_label['rating'].\
    apply(lambda x : changedict[x[0]] if isinstance(x,str) else None)
text_data_label.dropna(inplace=True)
'''
datalen=len(text_data_label)
text_data_label.dropna(inplace=True)
text_data_label=shuffle(text_data_label)
text_data_label.iloc[:int(datalen*0.7)].to_csv('./regression/londonreview_train.csv',index=False)
text_data_label.iloc[int(datalen*0.7)+1:].to_csv('./regression/londonreview_eval.csv',index=False)
unique=text_data_label['review_text'].values.tolist()
pass


