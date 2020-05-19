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
from scipy.stats import spearmanr

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
from lib.eval 			import print_nn_word, get_nn_list, get_cosine_similarity, get_pip_norm, get_word_relatedness
from lib.score 			import evaluate_analogy
from lib.operations 	import align, avg, join, align_list, get_common_vocab
from lib.util			import get_filename


#-------------------------------------------------------------------------------------------------------------------
# Experiments
#-------------------------------------------------------------------------------------------------------------------

languages = ['hi', 'fi', 'zh', 'cs', 'pl', 'pt', 'en']
languages = ['en']
models = ['glove', 'fasttext', 'word2vec']
model_types = {'word2vec': ['skipgram'], 'fasttext': ['skipgram'], 'glove': [None]}
data_types = ['shuffle']

total_run_num = 16	
target_word_num = int(1000)
n_values = [2,5,10,25,50]
n_max_value = max(n_values)

for language in languages:
	for model in models:
		model_type = model_types[model][0]
		data_type = data_types[0]

		if model == 'glove':
			folder = exp_folder + language + '/' + model + '/' + data_type
		else:
			folder = exp_folder + language + '/' + model + '/' + model_type + '/' + data_type


		# Get Common Vocabulary
		directory_list = [folder + '/run_{:04d}'.format(i) for i in range(total_run_num)]
		common_vocab = get_common_vocab(directory_list)

		m_base = Model(model)
		m_base.load(folder + '/run_{:04d}'.format(0))

		# Eval Indices	
		word_array = common_vocab
		np.random.shuffle(word_array)
		word_array = word_array[:target_word_num]

		p_at_n = np.zeros((target_word_num, len(n_values)))
		j_at_n = np.zeros((target_word_num, len(n_values)))

		# Array for storing the NNs:
		nn_array = np.zeros((total_run_num, target_word_num, n_max_value))

		# Load Models & Get NN Arrays
		for run_number in range(total_run_num):

			if run_number > 0:
				m = Model(model)
				m.load(folder + '/run_{:04d}'.format(run_number))
				__, m, ___ = align(m_base, m)
			else:
				m = m_base

			m = m.reduce(len(common_vocab))

			# Get NNs
			eval_indices = [m.indices[word] for word in word_array]
			nn_array[run_number] = get_nn_list(m, m, eval_indices, n_max_value, False, True)[0]

		#Analyse the overlap
		for model1_num in range(total_run_num):
			for model2_num in range(model1_num + 1, total_run_num):
				for target_word in range(target_word_num):
					for n_index in range(len(n_values)):
						n = n_values[n_index]

						word_list_1 = nn_array[model1_num,target_word,:n]
						word_list_2 = nn_array[model2_num,target_word,:n]

						p_at_n[target_word, n_index] += \
							len(np.intersect1d(word_list_1, word_list_2)) / n
						
						j_at_n[target_word, n_index] += \
							len(np.intersect1d(word_list_1,word_list_2)) / \
							len(np.union1d(word_list_1, word_list_2))

		p_at_n /= ((total_run_num - 1) * (total_run_num // 2))
		j_at_n /= ((total_run_num - 1) * (total_run_num // 2))

		p_result = list()
		print('Percentage Overlap:')
		for n_index_1 in range(len(n_values)):
			this_result = list()
			for n_index_2 in range(n_index_1 + 1, len(n_values)):
				res = spearmanr(p_at_n[:,n_index_1], p_at_n[:,n_index_2])[0]
				print(n_index_1, n_index_2, res)
				this_result.append(res)
			p_result.append(this_result)

		j_result = list()
		print('\nJaccard Coefficient:')
		for n_index_1 in range(len(n_values)):
			this_result = list()
			for n_index_2 in range(n_index_1 + 1, len(n_values)):
				res = spearmanr(j_at_n[:,n_index_1], j_at_n[:,n_index_2])[0]
				print(n_index_1, n_index_2, res)
				this_result.append(res)
			j_result.append(this_result)

		file_name = '/home/rettenls/code/eval/nn_metrics_evaluation/' + language + '_' + model + '.pkl'
		pickle.dump([p_result, j_result], open(file_name, 'wb'))