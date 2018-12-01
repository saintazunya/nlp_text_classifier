import tensorflow as tf
import pickle


class text_classifier():
    def __init__(self):
        self.sess = tf.Session()
        self.batch_size = 500
        self.learning_rate = 0.0001
        self.steps = 3000
        pass

    def load_data(self, train_path, datatype, eval_path=None, header=True, preprocessfunc=None):

        self.traindata = tf.contrib.data.CsvDataset(train_path, datatype, header=header, na_value='')
        self.traindata = self.traindata.repeat().shuffle(100, seed=1024).batch(self.batch_size,drop_remainder=True).map(preprocessfunc)
        self.iterator = tf.data.Iterator.from_structure(self.traindata.output_types,
                                                        self.traindata.output_shapes,
                                                        output_classes=self.traindata.output_classes)
        self.datatensor = self.iterator.get_next()
        self.training_init_op = self.iterator.make_initializer(self.traindata)
        if not eval_path:
            self.eval_init_op = self.iterator.make_initializer(self.traindata)
        else:
            self.evaldata = tf.contrib.data.CsvDataset(eval_path, datatype, header=header, na_value='')
            self.evaldata = self.evaldata.batch(self.batch_size,drop_remainder=True).map(preprocessfunc)
            self.eval_init_op = self.iterator.make_initializer(self.evaldata)
        #assert self.traindata.output_shapes==self.evaldata.output_shapes
        pass

    def test(self):
        pass

    def model(self):
        def extract_axis_1(data, ind):
            """
            Get specified elements along the first axis of tensor.
            :param data: Tensorflow tensor that will be subsetted.
            :param ind: Indices to take (one for each element along axis 0 of data).
            :return: Subsetted tensor.
            """

            batch_range = tf.range(tf.shape(data)[0])
            indices = tf.stack([batch_range, ind], axis=1)
            res = tf.gather_nd(data, indices)

            return res
        input_tensor = tf.contrib.feature_column.sequence_input_layer(self.datatensor,
                                                                      feature_columns=
                                                                      self.feature_columns[
                                                                          'text'])
        self.temp = input_tensor
        #input_tensor = tf.layers.batch_normalization(input_tensor, axis=1)
        input_tensor,seq_len=input_tensor
        input_tensor=tf.layers.batch_normalization(input_tensor)
        cell=tf.nn.rnn_cell.LSTMCell(100)
        initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(cell, input_tensor,
                                           initial_state=initial_state,
                                           sequence_length=seq_len,
                                           dtype=tf.float32)

        self.tempout=state
        self.output = tf.keras.layers.Dense(self.num_hash_bucket, activation='sigmoid',kernel_regularizer='l2')(
            state[1])

    def _compile(self):
        self.metric_dict = {}
        self.label_tensor = tf.feature_column.input_layer(self.datatensor,
                                                          feature_columns=self.feature_columns['label'])
        self.model_loss = tf.losses.sigmoid_cross_entropy(self.label_tensor, self.output)
        self.metric_dict['Entropy'] = self.model_loss
        self.k_best = 2
        k=self.k_best
        self.metric_dict['Top_{}_accuracy'.format(k)] = tf.metrics.average_precision_at_k(
            tf.cast(self.label_tensor, tf.int64), self.output, k)
        self.reg_loss = tf.losses.get_regularization_loss()
        self.loss = self.model_loss + self.reg_loss
        optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.gradient=optimizer.compute_gradients(self.loss)
        self.train_step = optimizer.minimize(self.loss)

        init_op = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        print('initalize the variables')
        self.sess.run(init_op)
        self.sess.run(init_l)

        with self.sess.as_default():
            print('Initize table')
            self.sess.run(tf.tables_initializer())
            ##print('Write tf graph')
            ##self.writer = tf.summary.FileWriter("./tmp", self.sess.graph)
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
                if key =='Top_{}_accuracy'.format(self.k_best):
                    resdict[key] = sum([x[key][0] for x in reslst]) / len(reslst)
                else:
                    resdict[key] = sum([x[key] for x in reslst]) / len(reslst)
            print('evalutation:', resdict)
            self.sess.run(self.training_init_op)
        pass

    def train(self):
        with self.sess.as_default():
            self.sess.run(self.training_init_op)
            for i in range(1, self.steps):
                step = self.sess.run([self.train_step, self.metric_dict])
                if not i % 25:
                    print(i, 'training:', step[1])
                    if not i % 100:
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

        text = tf.contrib.feature_column.sequence_categorical_column_with_vocabulary_list('text', vocablist.keys())
        text = tf.feature_column.embedding_column(text,50,initializer=init,trainable=False)
        self.num_hash_bucket = 20
        label = tf.feature_column.categorical_column_with_hash_bucket('label',20)
        label = tf.feature_column.indicator_column(label)

        self.feature_columns = dict(
            text=text,
            label=label,
        )
        pass


if __name__ == '__main__':
    tfmodel = text_classifier()
    preprocessfunc = lambda x, y: dict(zip(['text', 'label'], [tf.string_split(x), y]))
    tfmodel.load_data('../../data/lm/londonreview_train.csv', [tf.string, tf.string],
                      eval_path='../../data/lm/londonreview_train.csv',
                      header=True,
                      preprocessfunc=preprocessfunc)
    tfmodel.get_feature_columns()
    tfmodel.model()
    tfmodel._compile()
    tfmodel.train()
    pass
