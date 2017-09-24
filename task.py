from model_param_space import ModelParamSpace
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from optparse import OptionParser
from utils import logging_utils
import numpy as np
import pandas as pd
from utils.data_utils import load_dict_from_txt
import os
import config
import datetime
import tensorflow as tf
from refe import *

class AttrDict(dict):
	def __init__(self, *args, **kwargs):
		super(AttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self

class Task:
	def __init__(self, model_name, data_name, cv_runs, params_dict, logger):
		if data_name == "wn18":
			self.train_triples = pd.read_csv(config.WN18_TRAIN, names=["e1", "r", "e2"]).as_matrix()
			self.valid_triples = pd.read_csv(config.WN18_VALID, names=["e1", "r", "e2"]).as_matrix()
			self.test_triples = pd.read_csv(config.WN18_TEST, names=["e1", "r", "e2"]).as_matrix()
			self.e2id = load_dict_from_txt(config.WN18_E2ID)
			self.r2id = load_dict_from_txt(config.WN18_R2ID)
		elif data_name == "fb15k":
			self.train_triples = pd.read_csv(config.FB15K_TRAIN, names=["e1", "r", "e2"]).as_matrix()
			self.valid_triples = pd.read_csv(config.FB15K_VALID, names=["e1", "r", "e2"]).as_matrix()
			self.test_triples = pd.read_csv(config.FB15K_TEST, names=["e1", "r", "e2"]).as_matrix()
			self.e2id = load_dict_from_txt(config.FB15K_E2ID)
			self.r2id = load_dict_from_txt(config.FB15K_R2ID)
		elif data_name == "bp":
			self.train_triples = pd.read_csv(config.BP_TRAIN, names=["e1", "r", "e2"]).as_matrix()
			self.valid_triples = pd.read_csv(config.BP_VALID, names=["e1", "r", "e2"]).as_matrix()
			self.test_triples = pd.read_csv(config.BP_TEST, names=["e1", "r", "e2"]).as_matrix()
			self.e2id = load_dict_from_txt(config.BP_E2ID)
			self.r2id = load_dict_from_txt(config.BP_R2ID)
		elif data_name == "fb1m":
			self.train_triples = pd.read_csv(config.FB1M_TRAIN, names=["e1", "r", "e2"]).as_matrix()
			self.valid_triples = pd.read_csv(config.FB1M_VALID, names=["e1", "r", "e2"]).as_matrix()
			self.test_triples = pd.read_csv(config.FB1M_TEST, names=["e1", "r", "e2"]).as_matrix()
			self.e2id = load_dict_from_txt(config.FB1M_E2ID)
			self.r2id = load_dict_from_txt(config.FB1M_R2ID)
		else:
			raise AttributeError("Invalid data name! (Valid data name: wn18, fb15k, bp)")

		self.model_name = model_name
		self.data_name = data_name
		self.cv_runs = cv_runs
		self.params_dict = params_dict
		self.hparams = AttrDict(params_dict)
		self.logger = logger
		self.n_entities = len(self.e2id)
		self.n_relations = len(self.r2id)

		self.model = self._get_model() 
		self.saver = tf.train.Saver(tf.global_variables())
		checkpoint_path = os.path.abspath(config.CHECKPOINT_PATH)
		if not os.path.exists(checkpoint_path):
			os.makedirs(checkpoint_path)
		self.checkpoint_prefix = os.path.join(checkpoint_path, self.__str__())
	
	def __str__(self):
		return self.model_name

	def _get_model(self):
		args = [self.n_entities, self.n_relations, self.hparams]
		if "DistMult" in self.model_name:
			if "tanh" in self.model_name:
				return DistMult_tanh(*args)
			else:
				return DistMult(*args)
		else:
			raise AttributeError("Invalid model name! (Check model_param_space.py)")
	
	def _save(self, sess):
		path = self.saver.save(sess, self.checkpoint_prefix)
		print("Saved model to {}".format(path))
	
	def _print_param_dict(self, d, prefix="      ", incr_prefix="      "):
		for k, v in sorted(d.items()):
			if isinstance(v, dict):
				self.logger.info("%s%s:" % (prefix, k))
				self.print_param_dict(v, prefix+incr_prefix, incr_prefix)
			else:
				self.logger.info("%s%s: %s" % (prefix, k, v))
	
	def create_session(self):
		session_conf = tf.ConfigProto(
				intra_op_parallelism_threads=8,
				allow_soft_placement=True,
				log_device_placement=False)
		return tf.Session(config=session_conf)

	def cv(self):
		self.logger.info("=" * 50)
		self.logger.info("Params")
		self._print_param_dict(self.params_dict)
		self.logger.info("Results")
		self.logger.info("\t\tRun\t\tEpoch\t\tLoss\t\tAcc")

		cv_loss = []
		cv_acc = []
		for i in range(self.cv_runs):
			sess = self.create_session()
			sess.run(tf.global_variables_initializer())
			epoch, loss, acc = self.model.fit(sess, self.train_triples, self.valid_triples)
			self.logger.info("\t\t%d\t\t%d\t\t%f\t\t%f" % (i+1, epoch, loss, acc))
			cv_loss.append(loss)
			cv_acc.append(acc)
			sess.close()

		self.loss = np.mean(cv_loss)
		self.acc = np.mean(cv_acc)

		self.logger.info("CV Loss: %.3f" % self.loss)
		self.logger.info("CV Accuracy: %.3f" % self.acc)
		self.logger.info("-" * 50)
	
	def refit(self):
		sess = self.create_session()
		sess.run(tf.global_variables_initializer())
		self.model.fit(sess, np.concatenate((self.train_triples, self.valid_triples)))
		print("Evaluation:")
		self.model.validation(sess, self.test_triples)

		sess.close()
		return res

class TaskOptimizer:
	def __init__(self, model_name, data_name, max_evals, cv_runs, logger):
		self.model_name = model_name
		self.data_name = data_name
		self.max_evals = max_evals
		self.cv_runs = cv_runs
		self.logger = logger
		self.model_param_space = ModelParamSpace(self.model_name)

	def _obj(self, param_dict):
		param_dict = self.model_param_space._convert_into_param(param_dict)
		self.task = Task(self.model_name, self.data_name, self.cv_runs, param_dict, self.logger)
		self.task.cv()
		tf.reset_default_graph()
		ret = {
			"loss": -self.task.acc,
			"attachments": {
				"loss": self.task.loss,
			},
			"status": STATUS_OK
		}
		return ret

	def run(self):
		trials = Trials()
		best = fmin(self._obj, self.model_param_space._build_space(), tpe.suggest, self.max_evals, trials)
		best_params = space_eval(self.model_param_space._build_space(), best)
		best_params = self.model_param_space._convert_into_param(best_params)
		trial_loss = np.asarray(trials.losses(), dtype=float)
		best_ind = np.argmin(trial_loss)
		acc = -trial_loss[best_ind]
		loss = trials.trial_attachments(trials.trials[best_ind])["loss"]
		self.logger.info("-"*50)
		self.logger.info("Best CV Results:")
		self.logger.info("Loss: %.3f" % loss)
		self.logger.info("Accuracy: %.3f" % acc)
		self.logger.info("Best Param:")
		self.task._print_param_dict(best_params)
		self.logger.info("-"*50)

def parse_args(parser):
	parser.add_option("-m", "--model", type="string", dest="model_name", default="TransE_L2")
	parser.add_option("-d", "--data", type="string", dest="data_name", default="wn18")
	parser.add_option("-e", "--eval", type="int", dest="max_evals", default=100)
	parser.add_option("-c", "--cv", type="int", dest="cv_runs", default=3)
	options, args = parser.parse_args()
	return options, args

def main(options):
	time_str = datetime.datetime.now().isoformat()
	logname = "[Model@%s]_[Data@%s]_%s.log" % (
			options.model_name, options.data_name, time_str)
	logger = logging_utils._get_logger(config.LOG_PATH, logname)
	optimizer = TaskOptimizer(options.model_name, options.data_name, options.max_evals, options.cv_runs, logger)
	optimizer.run()

if __name__ == "__main__":
	parser = OptionParser()
	options, args = parse_args(parser)
	main(options)
