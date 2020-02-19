#-------------------------------------------------------------------------------------------------------------------
# Packages & Settings
#-------------------------------------------------------------------------------------------------------------------

# General packages
import time
import sys
import os
import datetime
from glob import glob

# Math and data structure packages
import numpy as np
from scipy import stats
import math
import random

# Plots, Fits, etc.
import matplotlib
import matplotlib.pyplot as plt


# Writing Output
import pickle

text_folder = '/home/rettenls/data/texts/wiki/'
eval_folder = '/home/rettenls/data/evaluation/analogy/'
exp_folder = '/home/rettenls/data/experiments/wiki/'
dist_folder = '/home/rettenls/data/experiments/wiki/analysis/distribution/'

coordination_file = exp_folder + 'coordination/coordinate.txt'

date_format = '%Y-%m-%d_%H:%M:%S'

#-------------------------------------------------------------------------------------------------------------------
# Loading own Modules
#-------------------------------------------------------------------------------------------------------------------

from lib.model 			import Model
from lib.trafo 			import Transformation
from lib.eval 			import print_nn_word, get_nn_list, get_cosine_similarity, get_pip_norm, get_word_relatedness
from lib.score 			import evaluate_analogy
from lib.operations 	import align, avg, join, align_list
from lib.util			import get_filename


#-------------------------------------------------------------------------------------------------------------------
# Experiments
#-------------------------------------------------------------------------------------------------------------------

languages = ['fi']#, 'cs', 'hi', 'zh', 'pt', 'pl', 'it']#, 'es', 'fr', 'de', 'en']
models = ['fasttext', 'glove', 'word2vec']
model_types = ['skipgram']
data_types = ['shuffle']#, 'bootstrap', 'fixed']

max_run_num = 128

for language in languages:
	for model in models:
		for model_type in model_types:
			for data_type in data_types:

				if model == 'glove':
					folder = exp_folder + language + '/' + model + '/' + data_type
				else:
					folder = exp_folder + language + '/' + model + '/' + model_type + '/' + data_type

				m = Model(model)
				m.load(folder + '/run_{:04d}'.format(0))

				eval_size = int(1.e4)
				max_eval_size = int(1.e3)

				# Eval Indices	
				eval_indices = np.arange(m.voc_size)
				np.random.shuffle(eval_indices)
				eval_indices = eval_indices[:eval_size]

				model_list = list()
				for j in range(2):
					m = Model(model)
					m.load(folder + '/run_{:04d}'.format(j))
					model_list.append(m)

				model_list = align_list(model_list)

				alignment_correct = True
				for j in range(10):
					index = random.randint(0, m.voc_size - 1)
					for l in range(1, len(model_list)):
						if (model_list[0].words[index] != model_list[l].words[index]):
							alignment_correct = False 

				print("Vocabularies aligned:" , alignment_correct, '\n')

				for proxy_size in [int(1.e3), int(1.e4), int(1.e5), int(1.e6)]:
					print('Proxy Size:', proxy_size)
					re_run_num = 16
					instability_eval = np.zeros((re_run_num, eval_size))
					for re_run in range(re_run_num):
						# Average Indices
						avg_indices = np.arange(m.voc_size)
						np.random.shuffle(avg_indices)
						avg_indices = avg_indices[:proxy_size]

						eval_steps = eval_size // max_eval_size

						instability_total = np.zeros(eval_size)
						counts_total = np.zeros(eval_size)

						for eval_step in range(eval_steps):

							lower = max_eval_size * eval_step
							upper = min(max_eval_size * (eval_step + 1), eval_size)

							eval_step_indices = eval_indices[lower:upper]
						
							pip_matrix = np.matmul(model_list[0].embeddings[eval_step_indices], model_list[0].embeddings[avg_indices].T) - \
											np.matmul(model_list[1].embeddings[eval_step_indices], model_list[1].embeddings[avg_indices].T) 
							instability = np.sqrt(np.sum(np.square(pip_matrix), axis = 1)) / math.sqrt(len(avg_indices))
							
							instability_total[lower:upper] += instability
							counts_total[lower:upper] += model_list[0].count[eval_step_indices] + model_list[1].count[eval_step_indices]

							print("Run:", re_run + 1, "of", re_run_num, ";", 
								upper, "of", eval_size, "words evaluated...", end = '\r')

						instability_eval[re_run] = instability_total

					print('')
					instability_means = np.mean(instability_eval, axis = 0)
					instability_std = np.std(instability_eval, axis = 0)
					instability_relative = instability_std / instability_means
					print("Proxy Size: {:.0E}, Average Relative Deviation of wwr PIP Loss: {:.1f}% \n".format(proxy_size, 100 * np.mean(instability_relative)))
