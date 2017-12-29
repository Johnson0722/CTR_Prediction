# coding:utf-8
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import tensorflow as tf
import numpy as np
from FFM.utilities import *
import math
import pandas as pd
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

class FFM(object):
    """
    Field-aware Factorization Machine
    """
    def __init__(self, config):
        """
        :param config: configuration of hyperparameters
        type of dict
        """
        # number of latent factors
        self.k = config['k']
        # num of fields
        self.f = config['f']
        # num of features
        self.p = feature_length
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.reg_l1 = config['reg_l1']
        self.reg_l2 = config['reg_l2']
        self.feature2field = config['feature2field']

    def add_placeholders(self):
        self.X = tf.placeholder('float32', [self.batch_size, self.p])
        self.y = tf.placeholder('int64', [None,])
        self.keep_prob = tf.placeholder('float32')

    def inference(self):
        """
        forward propagation
        :return: labels for each sample
        """
        with tf.variable_scope('linear_layer'):
            b = tf.get_variable('bias', shape=[2],
                                initializer=tf.zeros_initializer())
            w1 = tf.get_variable('w1', shape=[self.p, 2],
                                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            # shape of [None, 2]
            self.linear_terms = tf.add(tf.matmul(self.X, w1), b)

        with tf.variable_scope('field_aware_interaction_layer'):
            v = tf.get_variable('v', shape=[self.p, self.f, self.k], dtype='float32',
                                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            # shape of [None, 1]
            self.field_aware_interaction_terms = tf.constant(0, dtype='float32')
            # build dict to find f, key of feature,value of field
            for i in range(self.p):
                for j in range(i+1,self.p):
                    self.field_aware_interaction_terms += tf.multiply(
                        tf.reduce_sum(tf.multiply(v[i,self.feature2field[i]], v[j,self.feature2field[j]])),
                        tf.multiply(self.X[:,i], self.X[:,j])
                    )
        # shape of [None, 2]
        self.y_out = tf.add(self.linear_terms, self.field_aware_interaction_terms)
        self.y_out_prob = tf.nn.softmax(self.y_out)

    def add_loss(self):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_out)
            mean_loss = tf.reduce_mean(cross_entropy)
            self.loss = mean_loss
            tf.summary.scalar('loss', self.loss)

    def add_accuracy(self):
        # accuracy
        self.correct_prediction = tf.equal(tf.cast(tf.argmax(model.y_out,1), tf.int64), model.y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        # add summary to accuracy
        tf.summary.scalar('accuracy', self.accuracy)

    def train(self):
        # Applies exponential decay to learning rate
        self.global_step = tf.Variable(0, trainable=False)
        # define optimizer
        optimizer = tf.train.AdagradDAOptimizer(self.lr, global_step=self.global_step)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def build_graph(self):
        """build graph for model"""
        self.add_placeholders()
        self.inference()
        self.add_loss()
        self.add_accuracy()
        self.train()

def check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state("checkpoints")
    if ckpt and ckpt.model_checkpoint_path:
        logging.info("Loading parameters for the my CNN architectures...")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        logging.info("Initializing fresh parameters for the my Factorization Machine")

def train_model(sess, model, epochs=10, print_every=500):
    """training model"""
    # Merge all the summaries and write them out to train_logs
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train_logs', sess.graph)
    for e in range(epochs):
        num_samples = 0
        losses = []
        # get training data, iterable
        train_data = pd.read_csv('../avazu_CTR/train.csv', chunksize=model.batch_size)
        # batch_size data
        for data in train_data:
            actual_batch_size = len(data)
            batch_X = []
            batch_y = []
            for i in range(actual_batch_size):
                sample = data.iloc[i,:]
                array = one_hot_representation(sample, fields_train_dict, train_array_length)
                batch_X.append(array[:-2])
                batch_y.append(array[-1])
            batch_X = np.array(batch_X)
            batch_y = np.array(batch_y)
            # create a feed dictionary for this batch
            feed_dict = {model.X: batch_X, model.y: batch_y, model.keep_prob:1}
            loss, accuracy,  summary, global_step, _ = sess.run([model.loss, model.accuracy,
                                                                 merged,model.global_step,
                                                                 model.train_op], feed_dict=feed_dict)
            # aggregate performance stats
            losses.append(loss*actual_batch_size)

            num_samples += actual_batch_size
            # Record summaries and train.csv-set accuracy
            train_writer.add_summary(summary, global_step=global_step)
            # print training loss and accuracy
            if global_step % print_every == 0:
                logging.info("Iteration {0}: with minibatch training loss = {1} and accuracy of {2}"
                             .format(global_step, loss, accuracy))
                saver.save(sess, "checkpoints/model", global_step=global_step)
        # print loss of one epoch
        total_loss = np.sum(losses)/num_samples
        print("Epoch {1}, Overall loss = {0:.3g}".format(total_loss, e+1))



def test_model(sess, model, print_every = 50):
    """training model"""
    # get testing data, iterable
    test_data = pd.read_csv('/home/johnso/PycharmProjects/News_recommendation/CTR_prediction/avazu_CTR/test.csv',
                            chunksize=model.batch_size)
    test_step = 1
    # batch_size data
    for data in test_data:
        actual_batch_size = len(data)
        batch_X = []
        for i in range(actual_batch_size):
            sample = data.iloc[i,:]
            array = one_hot_representation(sample,fields_dict,test_array_length)
            batch_X.append(array)

        batch_X = np.array(batch_X)

        # create a feed dictionary for this batch
        feed_dict = {model.X: batch_X, model.keep_prob:1}
        # shape of [None,2]
        y_out_prob = sess.run([model.y_out_prob], feed_dict=feed_dict)
        # write to csv files
        data['click'] = y_out_prob[0][:,-1]
        if test_step == 1:
            data[['id','click']].to_csv('FM_FTRL_v1.csv', mode='a', index=False, header=True)
        else:
            data[['id','click']].to_csv('FM_FTRL_v1.csv', mode='a', index=False, header=False)

        test_step += 1
        if test_step % 50 == 0:
            logging.info("Iteration {0} has finished".format(test_step))


if __name__ == '__main__':
    '''launching TensorBoard: tensorboard --logdir=path/to/log-directory'''
    # seting fields
    fields= ['hour', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
                    'banner_pos', 'site_id' ,'site_domain', 'site_category', 'app_domain',
                    'app_id', 'app_category', 'device_model', 'device_type', 'device_id',
                    'device_conn_type','click']
    # loading dicts
    fields_dict = {}
    for field in fields:
        with open('dicts/'+field+'.pkl','rb') as f:
            fields_dict[field] = pickle.load(f)
    # loading feature2field dict
    with open('feature2field.pkl','rb') as f:
        feature2field = pickle.load(f)
    # length of representation
    train_array_length = max(fields_dict['click'].values()) + 1
    test_array_length = train_array_length - 2
    # initialize the model
    config = {}
    config['lr'] = 0.01
    config['batch_size'] = 512
    config['reg_l1'] = 2e-3
    config['reg_l2'] = 0
    config['k'] = 4
    config['f'] = len(fields) - 1
    config['feature2field'] = feature2field
    # get feature length
    feature_length = test_array_length
    # initialize FFM model
    model = FFM(config)
    # build graph for model
    model.build_graph()
    saver = tf.train.Saver(max_to_keep=5)
    with tf.Session() as sess:
        # TODO: with every epoches, print training accuracy and validation accuracy
        sess.run(tf.global_variables_initializer())
        # restore trained parameters
        check_restore_parameters(sess, saver)
        print('start training...')
        train_model(sess, model, epochs=10, print_every=50)
        # print('start validation...')
        # validation_model(sess, model, print_every=100)
        # print('start testing...')
        # test_model(sess, model, print_every=50)