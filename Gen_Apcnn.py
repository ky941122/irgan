# coding=utf-8
import tensorflow as tf
import numpy as np
import pickle
import time
#from QACNN import QACNN
from cnn import APCNN


class Generator(APCNN):

    def __init__(self, params, mode):
        APCNN.__init__(self, params, mode)
        self.model_type = "Gen"
        self.reward = tf.placeholder(tf.float32, shape=[None], name='reward')
        self.neg_index = tf.placeholder(tf.int32, shape=[None], name='neg_index')

        self.batch_scores = tf.nn.softmax(self.negative_score - self.score)  # ~~~~~
        # self.all_logits =tf.nn.softmax( self.score13) #~~~~~
        self.prob = tf.gather(self.batch_scores, self.neg_index)
        self.gan_loss = -tf.reduce_mean(tf.log(self.prob) * self.reward) # + l2_reg_lambda * self.l2_loss

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(params.learning_rate*0.1)
        grads_and_vars = optimizer.compute_gradients(self.gan_loss)
        self.gan_updates = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        # minize attention
        self.gan_score = self.negative_score - self.score
        self.dns_score = self.negative_score




