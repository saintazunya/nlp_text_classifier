import tensorflow as tf
from load_text_vector import loadGloveModel
import pickle

a=loadGloveModel('/home/zcf/Documents/Graduate/NLP/glove.twitter.27B.50d.txt')
pickle.dump(a,open('embedding_dict.pkl','wb'))
pass