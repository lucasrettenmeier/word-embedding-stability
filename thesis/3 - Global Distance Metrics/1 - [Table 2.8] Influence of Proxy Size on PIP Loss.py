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

sys.path.append('/home/rettenls/code')
from lib.model 			import Model
from lib.trafo 			import Transformation
from lib.eval 			import print_nn_word, get_nn_list, get_cosine_similarity, get_pip_norm, get_word_relatedness, get_common_vocab
from lib.score 			import evaluate_analogy
from lib.operations 	import align, avg, join, align_list
from lib.util			import get_filename
from scipy.stats		import spearmanr


#-------------------------------------------------------------------------------------------------------------------
# Experiments
#-------------------------------------------------------------------------------------------------------------------

languages = ['fi', 'hi', 'cs', 'zh', 'pt', 'pl', 'en']
models = ['word2vec']
model_type = 'skipgram'
data_type = 'shuffle'

target_word_num = int(1.e3)
total_pair_num = 10

for model in models:
	for language in languages:
		
		if model == 'glove':
			folder = exp_folder + language + '/' + model + '/' + data_type
		else:
			folder = exp_folder + language + '/' + model + '/' + model_type + '/' + data_type

		# Get Common Vocab
		directory_list = list()
		for i in range(total_pair_num * 2):
			directory_list.append(folder + '/run_{:04d}'.format(i))
		common_vocab = get_common_vocab(directory_list)

		# Get Target Words
		target_words = np.array(list(common_vocab))
		np.random.shuffle(target_words)
		target_words = target_words[:target_word_num]

		# Proxy Sizes
		proxy_sizes = [int(1.e3), int(1.e4), int(1.e5), len(common_vocab)]

		# Variable for Final Results
		spearman_technique = np.zeros((len(proxy_sizes), len(proxy_sizes)))
		spearman_all = np.zeros((total_pair_num, len(proxy_sizes), len(proxy_sizes)))

		# Load Models
		for model_pair_index in range(total_pair_num):
			m0 = Model(model)
			m0.load(folder + '/run_{:04d}'.format(model_pair_index * 2))
			m1 = Model(model)
			m1.load(folder + '/run_{:04d}'.format(model_pair_index * 2 + 1))
			m0,m1,joint = align(m0,m1)

			target_word_indices = [m0.indices[word] for word in target_words]

			pip_loss = np.zeros((2, len(proxy_sizes),target_word_num))
			for pip_loss_calc_run in range(2):
				for proxy_size_index in range(len(proxy_sizes)):

					# Get Proxy Size
					proxy_size = proxy_sizes[proxy_size_index]
					#print('Proxy Size:', proxy_size)

					# Randomly Draw Proxy Indices
					proxy_indices = np.arange(len(joint))
					np.random.shuffle(proxy_indices)
					proxy_indices = proxy_indices[:proxy_size]

					# Calculate wwrPIP Loss
					eval_step_size = max(int(1.e8) // proxy_size, 1) 
					eval_steps = target_word_num // eval_step_size
					for eval_step in range(eval_steps + 1):

						# Get Eval Indices
						lower = eval_step_size * eval_step
						upper = min(target_word_num, lower + eval_step_size)
						eval_step_indices = target_word_indices[lower:upper]
					
						# Evaluate Expression
						expression = (np.matmul(m0.embeddings[eval_step_indices], m0.embeddings[proxy_indices].T) - \
										np.matmul(m1.embeddings[eval_step_indices], m1.embeddings[proxy_indices].T)) / 2
						pip_loss_values = np.sqrt(np.sum(np.square(expression), axis = 1)) / math.sqrt(len(proxy_indices))
						pip_loss[pip_loss_calc_run, proxy_size_index, lower:upper] = pip_loss_values

			spearman = np.zeros((len(proxy_sizes), len(proxy_sizes)))
			for proxy_size_index_1 in range(len(proxy_sizes)):
				for proxy_size_index_2 in range(len(proxy_sizes)):
					spearman[proxy_size_index_1, proxy_size_index_2] = spearmanr(pip_loss[0,proxy_size_index_1],pip_loss[1,proxy_size_index_2])[0]

			print(spearman)
			spearman_all[model_pair_index] = spearman

		spearman_avg = np.sum(spearman_all, axis = 0)
		print(model, language)
		print(spearman_avg)
		print()	

		spearman_technique += spearman_avg

	spearman_technique /= (total_pair_num)
	print('\n--FINAL--')
	print(model)
	print(spearman_technique)
	print('--FINAL--\n')
				
				
					


		
