import tensorflow as tf
import numpy as np
import datetime
from utils import data_utils

class Model(object):
	def __init__(self, n_entities, n_relations, hparams):
		self.n_entities = n_entities
		self.n_relations = n_relations

		# required params
		self.embedding_size = hparams.embedding_size
		self.lr = hparams.lr
		self.batch_size = hparams.batch_size
		self.num_epochs = hparams.num_epochs

		# global step for tensorflow
		self.global_step = tf.Variable(0, name="global_step", trainable=False)

	def add_placeholders(self):
		self.heads = tf.placeholder(tf.int32, [None], name="head_entities")
		self.tails = tf.placeholder(tf.int32, [None], name="tail_entities")
		self.relations = tf.placeholder(tf.int32, [None], name="relations")
	
	def create_feed_dict(self, heads, relations, tails):
		feed_dict = {
			self.heads: heads,
			self.relations: relations,
			self.tails: tails,
		}
		return feed_dict

	def add_params(self):
		"""
		Define the variables in Tensorflow for params in the model.
		"""
		raise NotImplementedError("Each Model must re-implement this method.")

	def add_prediction_op(self):
		"""
		Define the prediction operator: self.pred.
		"""
		raise NotImplementedError("Each Model must re-implement this method.")

	def add_loss_op(self):
		"""
		Define the loss operator: self.loss.
		"""
		raise NotImplementedError("Each Model must re-implement this method.")

	def add_training_op(self):
		"""
		Define the training operator: self.train_op.
		"""
		optimizer = tf.train.AdamOptimizer(self.lr)
		self.grads_and_vars = optimizer.compute_gradients(self.loss)
		self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

	def train_on_batch(self, sess, input_batch):
		feed = self.create_feed_dict(**input_batch)
		_, step, loss, acc = sess.run([self.train_op, self.global_step, self.loss, self.accuracy], feed_dict=feed)
		time_str = datetime.datetime.now().isoformat()
		print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))
	
	def validation(self, sess, valid_triples):
		batches = data_utils.batch_iter(valid_triples, self.batch_size, 1, shuffle=False)
		total_loss = 0.0
		total_acc = 0.0
		total_len = 0
		for batch in batches:
			feed = self.create_feed_dict(**batch)
			loss, acc = sess.run([self.loss, self.accuracy], feed_dict=feed)
			total_loss += loss * len(batch["heads"])
			total_acc += acc * len(batch["heads"])
			total_len += len(batch["heads"])
		time_str = datetime.datetime.now().isoformat()
		print("{}: loss {:g} acc {:g}".format(time_str, total_loss/total_len, total_acc/total_len))
		return total_loss/total_len, total_acc/total_len
	
	def predict(self, sess, test_triples):
		batches = data_utils.batch_iter(test_triples, self.batch_size, 1, shuffle=False)
		preds = []
		for batch in batches:
			feed = self.create_feed_dict(**batch)
			pred = sess.run(self.preds, feed_dict=feed)
			preds = np.concatenate([preds, pred])
		return preds

	def fit(self, sess, train_triples, valid_triples=None):
		train_batches = data_utils.batch_iter(train_triples, self.batch_size, self.num_epochs)
		data_size = len(train_triples)
		num_batches_per_epoch = int((data_size-1)/self.batch_size) + 1
		best_valid_acc = 0.0
		best_valid_loss = 1e10
		best_valid_epoch = 0
		for batch in train_batches:
			self.train_on_batch(sess, batch)
			current_step = tf.train.global_step(sess, self.global_step)
			if (current_step % num_batches_per_epoch == 0) and (valid_triples is not None):
				print("\nValidation:")
				print("previous best valid epoch %d, best valid acc %.3f with loss %.3f" % (best_valid_epoch, best_valid_acc, best_valid_loss))
				loss, acc = self.validation(sess, valid_triples)
				print("")
				if acc > best_valid_acc:
					best_valid_loss = loss
					best_valid_acc = acc
					best_valid_epoch = current_step // num_batches_per_epoch
				if current_step//num_batches_per_epoch - best_valid_epoch >= 3:
					break
		return best_valid_epoch, best_valid_loss, best_valid_acc
	
	def build(self):
		self.add_placeholders()
		self.add_params()
		self.add_prediction_op()
		self.add_loss_op()
		self.add_training_op()
		tf.Graph().finalize()
