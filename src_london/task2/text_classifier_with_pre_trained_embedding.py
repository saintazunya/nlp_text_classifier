import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

class text_classifier():
    def __init__(self):
        self.sess = tf.Session()
        self.batch_size = 200
        self.learning_rate = 0.0001
        self.steps = 3000
        pass

    def load_data(self, train_path, datatype, eval_path=None, header=True, preprocessfunc=None):

        self.traindata = tf.contrib.data.CsvDataset(train_path, datatype, header=header, na_value='')
        self.traindata = self.traindata.repeat().shuffle(100, seed=1024).batch(self.batch_size).map(preprocessfunc)
        self.iterator = tf.data.Iterator.from_structure(self.traindata.output_types,
                                                        self.traindata.output_shapes,
                                                        output_classes=self.traindata.output_classes)
        self.datatensor = self.iterator.get_next()
        self.training_init_op = self.iterator.make_initializer(self.traindata)
        if not eval_path:
            self.eval_init_op = self.iterator.make_initializer(self.traindata)
        else:
            self.evaldata = tf.contrib.data.CsvDataset(eval_path, datatype, header=header, na_value='')
            self.evaldata = self.evaldata.batch(self.batch_size).map(preprocessfunc)
            self.eval_init_op = self.iterator.make_initializer(self.evaldata)
        pass

    def test(self):
        pass

    def model(self):
        input_tensor = tf.feature_column.input_layer(self.datatensor, feature_columns=self.feature_columns['text'])
        for i in range(2, 0, -1):
            input_tensor = tf.keras.layers.Dense(i * 30, kernel_regularizer='l2')(input_tensor)
        self.output = tf.keras.layers.Dense(1, kernel_regularizer='l2')(input_tensor)

    def _compile(self):
        self.metric_dict = {}
        self.label_tensor = tf.feature_column.input_layer(self.datatensor,
                                                          feature_columns=self.feature_columns['label'])
        self.metric_dict['MSE'] = tf.losses.mean_squared_error(self.label_tensor, self.output)
        self.reg_loss = tf.losses.get_regularization_loss()
        self.loss = self.metric_dict['MSE']+self.reg_loss
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        init_op = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        print('initalize the variables')
        self.sess.run(init_op)
        self.sess.run(init_l)

        with self.sess.as_default():
            print('Initize table')
            self.sess.run(tf.tables_initializer())
            #print('Write tf graph')
            #self.writer = tf.summary.FileWriter("./tmp", self.sess.graph)
            #self.writer.close()
        pass

    def eval_func(self):
        with self.sess.as_default():
            self.sess.run(self.eval_init_op)
            reslst = []
            resdict = {}
            while 1:
                try:
                    reslst.append(self.sess.run(self.metric_dict))
                except tf.errors.OutOfRangeError:
                    break
            for key in self.metric_dict:
                resdict[key] = sum([x[key] for x in reslst]) / len(reslst)
            print('evalutation:', resdict)
            self.sess.run(self.training_init_op)
        pass

    def train(self):
        with self.sess.as_default():
            self.sess.run(self.training_init_op)
            for i in range(1, self.steps):
                step = self.sess.run([self.train_step, self.metric_dict])
                if not i % 100:
                    print(i,'training:', step[1])
                    if not i % 500:
                        self.eval_func()

    def get_feature_columns(self):
        vocablist = pickle.load(open('../../data/vocablist.pkl', 'rb'))
        print(len(vocablist))
        init = tf.contrib.framework.load_embedding_initializer(
            ckpt_path='../../data/embedding_lookup',
            embedding_tensor_name='glove.twitter.27B.50d.txt',
            new_vocab_size=len(vocablist),
            embedding_dim=50,
            old_vocab_file='../../data/vocablist.txt',
            new_vocab_file='../../data/vocablist.txt'
        )

        text = tf.feature_column.categorical_column_with_vocabulary_list('text', vocablist.keys())
        text = tf.feature_column.embedding_column(text, 50, initializer=init, trainable=False)
        label = tf.feature_column.numeric_column('label')
        self.num_hash_bucket = len(vocablist.keys())
        self.feature_columns = dict(
            text=text,
            label=label,
        )
        pass


if __name__ == '__main__':
    tfmodel = text_classifier()
    preprocessfunc = lambda x, y: dict(zip(['text', 'label'], [tf.string_split(x), y]))
    tfmodel.load_data('../../data/regression/londonreview_train.csv', [tf.string, tf.float32],
                      eval_path='../../data/regression/londonreview_eval.csv', header=True,
                      preprocessfunc=preprocessfunc)
    tfmodel.get_feature_columns()
    tfmodel.model()
    tfmodel._compile()
    tfmodel.train()
    pass
