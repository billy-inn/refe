from model import Model
import tensorflow as tf
import datetime
import numpy as np
import config

class DistMult(Model):
	def __init__(self, n_entities, n_relations, hparams):
		super(DistMult, self).__init__(n_entities, n_relations, hparams)
		self.l2_reg_lambda = hparams.l2_reg_lambda
		self.build()

	def add_params(self):
		self.entity_embedding = tf.Variable(tf.random_uniform([self.n_entities, self.embedding_size], 0., 1., seed=config.RANDOM_SEED), dtype=tf.float32, name="entity_embedding")
		self.relation_embedding = tf.Variable(tf.random_uniform([self.n_relations, self.embedding_size], 0., 1., seed=config.RANDOM_SEED), dtype=tf.float32, name="relation_embedding")
	
	def add_prediction_op(self):
		self.e1 = tf.nn.embedding_lookup(self.entity_embedding, self.heads)
		self.e2 = tf.nn.embedding_lookup(self.entity_embedding, self.tails)

		self.scores = tf.matmul(self.e1 * self.e2, self.relation_embedding, transpose_b=True, name="scores")
		self.probs = tf.nn.softmax(self.scores, name="probs")
		self.preds = tf.argmax(self.probs, 1, name="preds")

		correct_predictions = tf.equal(tf.to_int32(self.preds), self.relations)
		self.accuracy = tf.reduce_mean(tf.to_float(correct_predictions), name="accuracy")

	def add_loss_op(self):
		losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.relations, logits=self.scores)
		self.l2_loss = tf.reduce_mean(tf.square(self.e1)) + \
				tf.reduce_mean(tf.square(self.e2)) + \
				tf.reduce_mean(tf.square(self.relation_embedding))
		self.loss = tf.add(tf.reduce_mean(losses), self.l2_reg_lambda * self.l2_loss, name="loss")

class DistMult_tanh(DistMult):
	def add_prediction_op(self):
		self.e1 = tf.tanh(tf.nn.embedding_lookup(self.entity_embedding, self.heads))
		self.e2 = tf.tanh(tf.nn.embedding_lookup(self.entity_embedding, self.tails))

		self.scores = tf.matmul(self.e1 * self.e2, self.relation_embedding, transpose_b=True, name="scores")
		self.probs = tf.nn.softmax(self.scores, name="probs")
		self.preds = tf.argmax(self.probs, 1, name="preds")

		correct_predictions = tf.equal(tf.to_int32(self.preds), self.relations)
		self.accuracy = tf.reduce_mean(tf.to_float(correct_predictions), name="accuracy")
