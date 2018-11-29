#coding=utf-8
from __future__ import division

import numpy as np
import os
import time
import datetime
import operator
import random
import tensorflow as tf
import pickle
import copy
import math

import Discriminator
import Generator
from insurance_qa_data_helpers import encode_sent
import insurance_qa_data_helpers
import data_loader
# import dataHelper

# Data
tf.flags.DEFINE_string("train_data", "data/id_data_sort", "train data (id)")
tf.flags.DEFINE_string("dev_data", "data/dev/id_dev_2w", "dev data (id)")
tf.flags.DEFINE_integer("vocab_size", 22511, "word vocab size")
# Model Hyperparameters
tf.flags.DEFINE_integer("data_every_round", 20000, "every round update Dis and Gen with how many data.")
tf.flags.DEFINE_integer("max_sequence_length", 16, "Max sequence length fo sentence (default: 200)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5,7,9", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 500, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.05, "learning_rate (default: 0.1)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 500, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 500000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 10, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("pools_size", 500, "The sampled set of a positive ample, which is bigger than 500")
tf.flags.DEFINE_integer("gen_pools_size", 20, "The sampled set of a positive ample, which is bigger than 500")
tf.flags.DEFINE_integer("g_epochs_num", 1, " the num_epochs of generator per epoch")
tf.flags.DEFINE_integer("d_epochs_num", 1, " the num_epochs of discriminator per epoch")
tf.flags.DEFINE_integer("sampled_size", 500, " the real selectd set from the The sampled pools")
tf.flags.DEFINE_integer("sampled_temperature", 20, " the temperature of sampling")
tf.flags.DEFINE_integer("gan_k", 5, "he number of samples of gan")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# Save Model
tf.flags.DEFINE_string("model_name", "model", "model name")
tf.flags.DEFINE_integer("num_checkpoints", 2000, "checkpoints number to save")
tf.flags.DEFINE_boolean("restore_model", False, "Whether restore model or create new parameters")
tf.flags.DEFINE_string("model_path", "runs", "Restore which model")
tf.flags.DEFINE_integer("init_loop", 0, "restore model from checkpoint and continue traing with former loop.")



FLAGS = tf.flags.FLAGS

#FLAGS._parse_flags()
# print(("\nParameters:"))
# for attr, value in sorted(FLAGS.__flags.items()):
#		 print(("{}={}".format(attr.upper(), value)))
# print((""))

timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time())))


print(("Loading data..."))


#vocab = insurance_qa_data_helpers.build_vocab()
# embeddings =insurance_qa_data_helpers.load_vectors(vocab)
alist = data_loader.read_alist(FLAGS.train_data, FLAGS.max_sequence_length)
raw = data_loader.read_raw(FLAGS.train_data, FLAGS.max_sequence_length)
assert len(raw) == len(alist)
print("Data number is:", len(raw))

ans = data_loader.read_ans(FLAGS.train_data, FLAGS.max_sequence_length)
dev_data = data_loader.read_dev(FLAGS.dev_data, FLAGS.max_sequence_length)


#test1List = insurance_qa_data_helpers.loadTestSet("test1")
#test2List= insurance_qa_data_helpers.loadTestSet("test2")
#devList= insurance_qa_data_helpers.loadTestSet("dev")
#testSet=[("test1",test1List),("test2",test2List),("dev",devList)]


print("Load done...")
log_precision = 'log/test1.gan_precision'+timeStamp
loss_precision = 'log/test1.gan_loss'+timeStamp

from functools import wraps
#print( tf.__version__)
def log_time_delta(func):
	@wraps(func)
	def _deco(*args, **kwargs):
		start = time.time()
		ret = func(*args, **kwargs)
		end = time.time()
		delta = end - start
		print( "%s runed %.2f seconds"% (func.__name__,delta))
		return ret
	return _deco



def generate_gan(sess, model, raw_loop, alist_loop, loss_type="pair",negative_size=3):
	samples=[]
	for _index ,pair in enumerate (raw_loop):
		if _index %100==0:
			print( "have sampled %d pairs" % _index)
		q=pair[0]
		a=pair[1]


		neg_alist_index=[i for i in range(len(alist_loop))]
		neg_alist_index.remove(_index)                 #remove the positive index

		simq = []
		head = _index - 1
		while head >= 0 and raw_loop[head][0] == q:
			simq.append(raw_loop[head][1])
			neg_alist_index.remove(head)
			head -= 1

		tail = _index + 1
		while tail < len(raw_loop) and raw_loop[tail][0] == q:
			simq.append(raw_loop[tail][1])
			neg_alist_index.remove(tail)
			tail += 1

		while head >= 0 and raw_loop[head][1] in simq:
			neg_alist_index.remove(head)
			head -= 1

		while tail < len(raw_loop) and raw_loop[tail][1] in simq:
			neg_alist_index.remove(tail)
			tail += 1

		sampled_index=np.random.choice(neg_alist_index,size=[FLAGS.pools_size],replace= False)
		pools=np.array(alist_loop)[sampled_index]

		canditates=insurance_qa_data_helpers.loadCandidateSamples(q,a,pools)
		predicteds=[]
		for batch in insurance_qa_data_helpers.batch_iter(canditates,batch_size=FLAGS.batch_size):							
			feed_dict = {model.input_x_1: batch[:,0],model.input_x_2: batch[:,1],model.input_x_3: batch[:,2]}			
			predicted=sess.run(model.gan_score,feed_dict)
			predicteds.extend(predicted)

		# index=np.argmax(predicteds)
		# samples.append([encode_sent(vocab,item, FLAGS.max_sequence_length) for item in [q,a,pools[index]]])
		exp_rating = np.exp(np.array(predicteds)*FLAGS.sampled_temperature*1.5)
		prob = exp_rating / np.sum(exp_rating)
		neg_samples_index = np.random.choice(np.arange(len(pools)), size= negative_size,p=prob,replace=False)
		for neg in neg_samples_index:
			samples.append([q, a, pools[neg]])
	return samples


@log_time_delta	 
def dev_step(sess,model,dev_data,k=40):
	cnt = 0
	dev_count = 0
	for userq in dev_data:
		print "\tEvaluation step:", dev_count
		dev_count += 1

		q = userq.strip().split()
		q = q[:FLAGS.max_sequence_length]
		q = q + [1] * (FLAGS.max_sequence_length - len(q))
		qs = []
		for a in ans:
			qs.append(q)
		feed_dict = {
			model.input_x_1: qs,
			model.input_x_2: ans,
			model.input_x_3: ans
		}

		scorel2 = tf.reshape(model.score12, [-1])
		topk = tf.nn.top_k(scorel2, k)

		index = sess.run(topk, feed_dict)[1]

		recalls = np.array(ans)[index]   #召回的相似Q
		for recall in recalls:
			recall = list(recall)
			if recall in dev_data[userq]:
				cnt += 1
				break       #有一个相似命中了就退出

	return cnt / len(dev_data)


@log_time_delta	 
def evaluation(sess,model,log, dev_data, num_epochs=0):
	current_step = tf.train.global_step(sess, model.global_step)
	if isinstance(model,  Discriminator.Discriminator):
		model_type="Dis"
	else:
		model_type="Gen"

	precision_current=dev_step(sess,model,dev_data,40)
	line="test1: %d epoch: precision %f"%(current_step,precision_current)
	print (line)
	print( model.save_model(sess,precision_current))
	log.write(line+"\n")
	log.flush()


	
def main():
	with tf.Graph().as_default():
		with tf.device("/gpu:1"):
			session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,log_device_placement=FLAGS.log_device_placement)
			session_conf.gpu_options.allow_growth = True
			sess = tf.Session(config=session_conf)

			with sess.as_default() ,open(log_precision,"w") as log,open(loss_precision,"w") as loss_log :
		
				param= None
				loss_type="pair"
				discriminator = Discriminator.Discriminator(
						sequence_length=FLAGS.max_sequence_length,
						batch_size=FLAGS.batch_size,
						vocab_size=FLAGS.vocab_size,
						embedding_size=FLAGS.embedding_dim,
						filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
						num_filters=FLAGS.num_filters,
						learning_rate=FLAGS.learning_rate,
						l2_reg_lambda=FLAGS.l2_reg_lambda,
						# embeddings=embeddings,
						embeddings=None,
						paras=param,
						loss=loss_type)

				generator = Generator.Generator(
						sequence_length=FLAGS.max_sequence_length,
						batch_size=FLAGS.batch_size,
						vocab_size=FLAGS.vocab_size,
						embedding_size=FLAGS.embedding_dim,
						filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
						num_filters=FLAGS.num_filters,
						learning_rate=FLAGS.learning_rate*0.1,
						l2_reg_lambda=FLAGS.l2_reg_lambda,
						# embeddings=embeddings,
						embeddings=None,
						paras=param,
						loss=loss_type)

				timestamp = str(int(time.time()))
				out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.model_name, timestamp))
				print("Writing to {}\n".format(out_dir))
				checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
				checkpoint_prefix = os.path.join(checkpoint_dir, "model")
				checkpoint_prefix_Dis = os.path.join(checkpoint_dir, "DIS_model")
				if not os.path.exists(checkpoint_dir):
					os.makedirs(checkpoint_dir)
				saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

				sess.run(tf.global_variables_initializer())

				restore = FLAGS.restore_model
				if restore:
					saver.restore(sess, FLAGS.model_path)
					print "*" * 20 + "\nReading model parameters from %s \n" % FLAGS.model_path + "*" * 20
				else:
					print "*" * 20 + "\nCreated model with fresh parameters.\n" + "*" * 20



				# evaluation(sess,discriminator,log,0)
				num_round_per_epoch = int(math.ceil(len(raw)/FLAGS.data_every_round))
				print "One epoch has", num_round_per_epoch, "loops."
				for i in range(FLAGS.num_epochs):
					if i == 0:
						loop = FLAGS.init_loop
					else:
						loop = 0
					while loop < num_round_per_epoch:
						end_index = min((loop + 1) * FLAGS.data_every_round, len(raw))
						raw_loop = raw[end_index-FLAGS.data_every_round:end_index]
						alist_loop = alist[end_index-FLAGS.data_every_round:end_index]
						assert len(raw_loop) == len(alist_loop)

						if loop>0 or i>0:
							samples=generate_gan(sess,generator, raw_loop, alist_loop)
							# for j in range(FLAGS.d_epochs_num):
							for _index,batch in enumerate(insurance_qa_data_helpers.batch_iter(samples,num_epochs=FLAGS.d_epochs_num,batch_size=FLAGS.batch_size,shuffle=True)):	# try:

								feed_dict = {discriminator.input_x_1: batch[:,0],discriminator.input_x_2: batch[:,1],discriminator.input_x_3: batch[:,2]}
								_, step,	current_loss,accuracy = sess.run(
										[discriminator.train_op, discriminator.global_step, discriminator.loss,discriminator.accuracy],
										feed_dict)

								line=("%s: DIS step %d, loss %f with acc %f "%(datetime.datetime.now().isoformat(), step, current_loss,accuracy))
								if _index%10==0:
									print(line)
								loss_log.write(line+"\n")
								loss_log.flush()

							if loop != 0 and loop % 20 == 0:
								evaluation(sess,discriminator,log, dev_data, i)

							path = saver.save(sess, checkpoint_prefix_Dis, global_step=generator.global_step)
							print("Saved DIS model checkpoint to {}\n".format(path))

						for g_epoch in range(FLAGS.g_epochs_num):
							for _index,pair in enumerate(raw_loop):

								q=pair[0]
								a=pair[1]

								neg_alist_index=[item for item in range(len(alist_loop))]
								neg_alist_index.remove(_index)  # remove the positive index

								simq = []
								head = _index - 1
								while head >= 0 and raw_loop[head][0] == q:
									simq.append(raw_loop[head][1])
									neg_alist_index.remove(head)
									head -= 1

								tail = _index + 1
								while tail < len(raw_loop) and raw_loop[tail][0] == q:
									simq.append(raw_loop[tail][1])
									neg_alist_index.remove(tail)
									tail += 1

								while head >= 0 and raw_loop[head][1] in simq:
									neg_alist_index.remove(head)
									head -= 1

								while tail < len(raw_loop) and raw_loop[tail][1] in simq:
									neg_alist_index.remove(tail)
									tail += 1

								sampled_index=np.random.choice(neg_alist_index,size=[FLAGS.pools_size-1],replace= False)
								sampled_index=list(sampled_index)
								sampled_index.append(_index)
								pools=np.array(alist_loop)[sampled_index]

								samples=insurance_qa_data_helpers.loadCandidateSamples(q,a,pools)
								predicteds=[]
								for batch in insurance_qa_data_helpers.batch_iter(samples,batch_size=FLAGS.batch_size):
									feed_dict = {generator.input_x_1: batch[:,0],generator.input_x_2: batch[:,1],generator.input_x_3: batch[:,2]}

									predicted=sess.run(generator.gan_score,feed_dict)
									predicteds.extend(predicted)

								exp_rating = np.exp(np.array(predicteds)*FLAGS.sampled_temperature)
								prob = exp_rating / np.sum(exp_rating)

								neg_index = np.random.choice(np.arange(len(pools)) , size=FLAGS.gan_k, p=prob ,replace=False)	# 生成 FLAGS.gan_k个负例

								subsamples=np.array(insurance_qa_data_helpers.loadCandidateSamples(q,a,pools[neg_index]))
								feed_dict = {discriminator.input_x_1: subsamples[:,0],discriminator.input_x_2: subsamples[:,1],discriminator.input_x_3: subsamples[:,2]}
								reward = sess.run(discriminator.reward,feed_dict)				 # reward= 2 * (tf.sigmoid( score_13 ) - 0.5)

								samples=np.array(samples)
								feed_dict = {
												generator.input_x_1: samples[:,0],
												generator.input_x_2: samples[:,1],
												generator.neg_index: neg_index,
												generator.input_x_3: samples[:,2],
												generator.reward: reward
											}
								_, step,	current_loss,positive,negative = sess.run(																					#应该是全集上的softmax	但是此处做全集的softmax开销太大了
										[generator.gan_updates, generator.global_step, generator.gan_loss, generator.positive,generator.negative],		 #	 self.prob= tf.nn.softmax( self.cos_13)
										feed_dict)																													#self.gan_loss = -tf.reduce_mean(tf.log(self.prob) * self.reward)

								line=("%s: Epoch %d, Loop %d, GEN step %d, loss %f  positive %f negative %f"%(datetime.datetime.now().isoformat(), i, loop, step, current_loss,positive,negative))
								if _index %10==0:
									print(line)
								loss_log.write(line+"\n")
								loss_log.flush()

							if loop != 0 and loop % 20 == 0:
								evaluation(sess,generator,log, dev_data, i*FLAGS.g_epochs_num + g_epoch)
							log.flush()

						path = saver.save(sess, checkpoint_prefix, global_step=generator.global_step)
						print("Saved model checkpoint to {}\n".format(path))
						loop += 1



										
if __name__ == '__main__':

	main()

