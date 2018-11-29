# coding=utf-8
import tensorflow as tf
import numpy as np
import time
import pickle
#from QACNN import QACNN
from cnn import APCNN


class Discriminator(APCNN):

    def __init__(self, params, mode):
        APCNN.__init__(self, params, mode)
        self.model_type = "Dis"

        with tf.name_scope("output"):
            self.losses = tf.maximum(0.0, tf.subtract(0.05, tf.subtract(self.score, self.negative_score)))
            self.loss = tf.reduce_sum(self.losses) # + self.l2_reg_lambda * self.l2_loss

            self.reward = 2.0 * (tf.sigmoid(tf.subtract(0.05, tf.subtract(self.score, self.negative_score))) - 0.5)  # no log
            self.positive = tf.reduce_mean(self.score)
            self.negative = tf.reduce_mean(self.negative_score)

            self.correct = tf.equal(0.0, self.losses)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
        self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)




