import tensorflow as tf
import pickle
import pandas as pd
import numpy as np


def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile, 'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


def load_embedding(embedding_path, vocab):
    model = loadGloveModel(embedding_path)
    res = []
    for v in vocab:
        res.append(model[v])
    return np.array(res),model
    pass


if __name__ == '__main__':
    sess = tf.Session()
    golve_path = '/home/zcf/Documents/Graduate/NLP/glove.twitter.27B.50d.txt'

    #module to save the embedding
    
 
    data=pickle.load(open('./data/vocablist.pkl','rb'))
    vocablist=list(data.keys())
    embedding_weights,_=load_embedding(golve_path,vocablist)
    const=tf.constant(embedding_weights,name="tf_var_initialized_from_np",dtype=tf.float32)
    tf.get_variable('glove.twitter.27B.50d.txt',initializer=const)
    init_op = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    sess.run(init_op)
    sess.run(init_l)
    saver = tf.train.Saver()
    saver.save(sess, '/home/zcf/Documents/Graduate/NLP/nlp_text_classifier/London_Rest_reviews/data/embedding_lookup')
    '''

    data = pickle.load(open('./data/vocablist.pkl', 'rb'))
    vocablist = list(data.keys())
    vocabfile=open('./data/vocablist.txt','w')
    [vocabfile.write(x+'\n') for x in vocablist]
    vocabfile.close()
    data = pd.read_csv('./data/londonreview_train.csv').iloc[0:100].review_text.values
    rawdata=data.tolist()[0].split()
    data = tf.data.Dataset.from_tensor_slices({'test': [[x] for x in rawdata]})
    data = data.make_one_shot_iterator().get_next()
    a = tf.feature_column.categorical_column_with_vocabulary_list('test', vocablist)
    init=tf.contrib.framework.load_embedding_initializer(
    ckpt_path='/home/zcf/Documents/Graduate/NLP/nlp_text_classifier/London_Rest_reviews/data/embedding_lookup',
    embedding_tensor_name='glove.twitter.27B.50d.txt',
    new_vocab_size=len(vocablist),
    embedding_dim=50,
    old_vocab_file='./data/vocablist.txt',
    new_vocab_file='./data/vocablist.txt'
        )
    b = tf.feature_column.embedding_column(a,50,initializer=init,trainable=False
            )

    # b=tf.feature_column.indicator_column(a)
    #data=tf.string('.')
    c = tf.feature_column.input_layer(data, b)
    init_op = tf.global_variables_initializer()
    embedding_weights,model = load_embedding(golve_path, vocablist)
    with sess.as_default():
        sess.run(init_op)
        sess.run(tf.tables_initializer())
        for x in rawdata:
            embedding_from_ftcl=sess.run(c)
            embedding_from_model=model[x]
    '''
    pass
