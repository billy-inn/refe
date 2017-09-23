import numpy as np
from hyperopt import hp

param_space_DistMult = {
	"embedding_size": hp.quniform("embedding_size", 50, 200, 10),
	"l2_reg_lambda": hp.qloguniform("l2_reg_lambda", np.log(1e-4), np.log(5e-2), 1e-4),
	"lr": hp.qloguniform("lr", np.log(1e-3), np.log(1e-1), 1e-3),
	"batch_size": 5000,
	"num_epochs": 20,
}

param_space_dict = {
	"DistMult": param_space_DistMult,
	"DistMult_tanh": param_space_DistMult,
}

int_params = [
	"embedding_size", "batch_size", "max_iter", "neg_ratio", "valid_every", "k",
]

class ModelParamSpace:
	def __init__(self, learner_name):
		s = "Invalid model name! (Check model_param_space.py)"
		assert learner_name in param_space_dict, s
		self.learner_name = learner_name
	
	def _build_space(self):
		return param_space_dict[self.learner_name]

	def _convert_into_param(self, param_dict):
		if isinstance(param_dict, dict):
			for k,v in param_dict.items():
				if k in int_params:
					param_dict[k] = int(v)
				elif isinstance(v, list) or isinstance(v, tuple):
					for i in range(len(v)):
						self._convert_into_param(v[i])
				elif isinstance(v, dict):
					self._convert_into_param(v)
		return param_dict
