# coding:utf-8
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import tensorflow as tf
import numpy as np
from Deep_FM.utilities import *
import math
import pandas as pd
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

class DeepFM(object):
    """
    Deep FM with FTRL optimization
    """
    def __init__(self, config):
        """
        :param config: configuration of hyperparameters
        type of dict
        """
        # number of latent factors
        self.k = config['k']
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.reg_l1 = config['reg_l1']
        self.reg_l2 = config['reg_l2']
        # num of features
        self.p = feature_length
        # num of fields
        self.field_cnt = field_cnt

    def add_placeholders(self):
        self.X = tf.placeholder('float32', [None, self.p])
        self.y = tf.placeholder('int64', [None,])
        # index of none-zero features
        self.feature_inds = tf.placeholder('int64', [None,field_cnt])
        self.keep_prob = tf.placeholder('float32')

    def inference(self):
        """
        forward propagation
        :return: labels for each sample
        """
        v = tf.Variable(tf.truncated_normal(shape=[self.p, self.k], mean=0, stddev=0.01),dtype='float32')

        # Factorization Machine
        with tf.variable_scope('FM'):
            b = tf.get_variable('bias', shape=[2],
                                initializer=tf.zeros_initializer())
            w1 = tf.get_variable('w1', shape=[self.p, 2],
                                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            # shape of [None, 2]
            self.linear_terms = tf.add(tf.matmul(self.X, w1), b)

            # shape of [None, 1]
            self.interaction_terms = tf.multiply(0.5,
                                                 tf.reduce_mean(
                                                     tf.subtract(
                                                         tf.pow(tf.matmul(self.X, v), 2),
                                                         tf.matmul(tf.pow(self.X, 2), tf.pow(v, 2))),
                                                     1, keep_dims=True))
            # shape of [None, 2]
            self.y_fm = tf.add(self.linear_terms, self.interaction_terms)

        # three-hidden-layer neural network, network shape of (200-200-200)
        with tf.variable_scope('DNN',reuse=False):
            # embedding layer
            y_embedding_input = tf.reshape(tf.gather(v, self.feature_inds), [-1, self.field_cnt*self.k])
            # first hidden layer
            w1 = tf.get_variable('w1_dnn', shape=[self.field_cnt*self.k, 200],
                                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            b1 = tf.get_variable('b1_dnn', shape=[200],
                                 initializer=tf.constant_initializer(0.001))
            y_hidden_l1 = tf.nn.relu(tf.matmul(y_embedding_input, w1) + b1)
            # second hidden layer
            w2 = tf.get_variable('w2', shape=[200, 200],
                                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            b2 = tf.get_variable('b2', shape=[200],
                                 initializer=tf.constant_initializer(0.001))
            y_hidden_l2 = tf.nn.relu(tf.matmul(y_hidden_l1, w2) + b2)
            # third hidden layer
            w3 = tf.get_variable('w1', shape=[200, 200],
                                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            b3 = tf.get_variable('b1', shape=[200],
                                 initializer=tf.constant_initializer(0.001))
            y_hidden_l3 = tf.nn.relu(tf.matmul(y_hidden_l2, w3) + b3)
            # output layer
            w_out = tf.get_variable('w_out', shape=[200, 2],
                                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            b_out = tf.get_variable('b_out', shape=[2],
                                 initializer=tf.constant_initializer(0.001))
            self.y_dnn = tf.nn.relu(tf.matmul(y_hidden_l3, w_out) + b_out)
        # add FM output and DNN output
        self.y_out = tf.add(self.y_fm, self.y_dnn)
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
        optimizer = tf.train.FtrlOptimizer(self.lr, l1_regularization_strength=self.reg_l1,
                                           l2_regularization_strength=self.reg_l2)
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
    num_samples = 0
    losses = []
    # Merge all the summaries and write them out to train_logs
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train_logs', sess.graph)
    for e in range(epochs):
        # get training data, iterable
        train_data = pd.read_csv('../avazu_CTR/train.csv',
                                 chunksize=model.batch_size)
        # batch_size data
        for data in train_data:
            actual_batch_size = len(data)
            batch_X = []
            batch_y = []
            batch_idx = []
            for i in range(actual_batch_size):
                sample = data.iloc[i,:]
                array,idx = one_hot_representation(sample,fields_train_dict, train_array_length)
                batch_X.append(array[:-2])
                batch_y.append(array[-1])
                batch_idx.append(idx)
            batch_X = np.array(batch_X)
            batch_y = np.array(batch_y)
            batch_idx = np.array(batch_idx)
            # create a feed dictionary for this batch
            feed_dict = {model.X: batch_X, model.y: batch_y,
                         model.feature_inds: batch_idx, model.keep_prob:1}
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

def validation_model(sess, model, print_every=50):
    """testing model"""
    # num samples
    num_samples = 0
    # num of correct predictions
    num_corrects = 0
    losses = []
    # Merge all the summaries and write them out to train_logs
    merged = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter('test_logs', sess.graph)
    # get testing data, iterable
    validation_data = pd.read_csv('/home/katy/CTR_prediction/avazu_CTR/train.csv',
                                  chunksize=model.batch_size)
    # testing step
    valid_step = 1
    # batch_size data
    for data in validation_data:
        actual_batch_size = len(data)
        batch_X = []
        batch_y = []
        batch_idx = []
        for i in range(actual_batch_size):
            sample = data.iloc[i,:]
            array,idx = one_hot_representation(sample,fields_train_dict, train_array_length)
            batch_X.append(array[:-2])
            batch_y.append(array[-1])
            batch_idx.append(idx)
        batch_X = np.array(batch_X)
        batch_y = np.array(batch_y)
        batch_idx = np.array(batch_idx)
        # create a feed dictionary for this batch,
        feed_dict = {model.X: batch_X, model.y: batch_y,
                 model.feature_inds: batch_idx, model.keep_prob:1}
        loss, accuracy, correct, summary = sess.run([model.loss, model.accuracy,
                                                     model.correct_prediction, merged,],
                                                    feed_dict=feed_dict)
        # aggregate performance stats
        losses.append(loss*actual_batch_size)
        num_corrects += correct
        num_samples += actual_batch_size
        # Record summaries and train.csv-set accuracy
        test_writer.add_summary(summary, global_step=valid_step)
        # print training loss and accuracy
        if valid_step % print_every == 0:
            logging.info("Iteration {0}: with minibatch training loss = {1} and accuracy of {2}"
                         .format(valid_step, loss, accuracy))
        valid_step += 1
    # print loss and accuracy of one epoch
    total_correct = num_corrects/num_samples
    total_loss = np.sum(losses)/num_samples
    print("Overall test loss = {0:.3g} and accuracy of {1:.3g}" \
          .format(total_loss,total_correct))


def test_model(sess, model, print_every = 50):
    """training model"""
    # get testing data, iterable
    test_data = pd.read_csv('/home/katy/CTR_prediction/avazu_CTR/test.csv',
                            chunksize=model.batch_size)
    test_step = 1
    # batch_size data
    for data in test_data:
        actual_batch_size = len(data)
        batch_X = []
        batch_idx = []
        for i in range(actual_batch_size):
            sample = data.iloc[i,:]
            array,idx = one_hot_representation(sample, fields_test_dict, test_array_length)
            batch_X.append(array)
            batch_idx.append(idx)

        batch_X = np.array(batch_X)
        batch_idx = np.array(batch_idx)
        # create a feed dictionary for this batch
        feed_dict = {model.X: batch_X, model.keep_prob:1, model.feature_inds:batch_idx}
        # shape of [None,2]
        y_out_prob = sess.run([model.y_out_prob], feed_dict=feed_dict)
        # write to csv files
        data['click'] = y_out_prob[0][:,-1]
        if test_step == 1:
            data[['id','click']].to_csv('Deep_FM_FTRL_v1.csv', mode='a', index=False, header=True)
        else:
            data[['id','click']].to_csv('Deep_FM_FTRL_v1.csv', mode='a', index=False, header=False)

        test_step += 1
        if test_step % 50 == 0:
            logging.info("Iteration {0} has finished".format(test_step))




if __name__ == '__main__':
    '''launching TensorBoard: tensorboard --logdir=path/to/log-directory'''
    # seting fields
    fields_train = ['hour', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
                    'banner_pos', 'site_id' ,'site_domain', 'site_category', 'app_domain',
                    'app_id', 'app_category', 'device_model', 'device_type', 'device_id',
                    'device_conn_type','click']

    fields_test = ['hour', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
                   'banner_pos', 'site_id' ,'site_domain', 'site_category', 'app_domain',
                   'app_id', 'device_id', 'app_category', 'device_model', 'device_type',
                   'device_conn_type']
    # loading dicts
    fields_train_dict = {}
    for field in fields_train:
        with open('dicts/'+field+'.pkl','rb') as f:
            fields_train_dict[field] = pickle.load(f)
    fields_test_dict = {}
    for field in fields_test:
        with open('dicts/'+field+'.pkl','rb') as f:
            fields_test_dict[field] = pickle.load(f)

    # length of representation
    train_array_length = max(fields_train_dict['click'].values()) + 1
    test_array_length = train_array_length - 2
    # initialize the model
    config = {}
    config['lr'] = 0.01
    config['batch_size'] = 512
    config['reg_l1'] = 2e-3
    config['reg_l2'] = 0
    config['k'] = 40
    # get feature length
    feature_length = test_array_length
    # num of fields
    field_cnt = 21

    model = DeepFM(config)
    # build graph for model
    model.build_graph()

    saver = tf.train.Saver(max_to_keep=5)


    with tf.Session() as sess:
        # TODO: with every epoches, print training accuracy and validation accuracy
        sess.run(tf.global_variables_initializer())
        # restore trained parameters
        check_restore_parameters(sess, saver)
        print('start training...')
        train_model(sess, model, epochs=10, print_every=500)
        # print('start validation...')
        # validation_model(sess, model, print_every=100)
        # print('start testing...')
        # test_model(sess, model)