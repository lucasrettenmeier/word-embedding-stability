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

sys.path.append('../../')
from lib.model 			import Model
from lib.trafo 			import Transformation
from lib.eval 			import print_nn_word, get_nn_list, get_cosine_similarity, get_pip_norm, get_word_relatedness
from lib.score 			import evaluate_analogy
from lib.operations 	import align, avg, join, align_list, get_common_vocab
from lib.util			import get_filename


#-------------------------------------------------------------------------------------------------------------------
# Experiments
#-------------------------------------------------------------------------------------------------------------------

languages = ['en']#, 'fi', 'zh', 'cs', 'pl', 'pt', 'en']
models = ['word2vec']#, 'fasttext', 'glove']
model_types = {'word2vec': ['cbow'], 'fasttext': ['skipgram'], 'glove': [None]}
data_types = ['shuffle']

total_run_num = 40

for language in languages:
	for model in models:
		model_type = model_types[model][0]
		data_type = data_types[0]

		if model == 'glove':
				folder = exp_folder + language + '/' + model + '/' + data_type
		else:
			folder = exp_folder + language + '/' + model + '/' + model_type + '/' + data_type

		m_base = Model(model)
		m_base.load(folder + '/run_{:04d}'.format(0))
		m_base = m_base.reduce(200000)


		# Query Word Indices
		query_word_index_list = get_nn_list(m_base,m_base, [m_base.indices['momentum']], 1000)[0][0]
		query_word_list = [m_base.words[i] for i in query_word_index_list]
		query_word_num = len(query_word_list)

		# Array for storing the COS-SIMs:
		cs_array = np.zeros((total_run_num, query_word_num))

		# Load Models & Get NN Arrays
		for run_number in range(total_run_num):

			try:
				if run_number > 0:
					m = Model(model)
					m.load(folder + '/run_{:04d}'.format(run_number))
				else:
					m = m_base

				m = m.reduce(200000)

				target_word_indices = np.array([m.indices['momentum']])
				query_word_indices = np.array([m.indices[w] for w in query_word_list])

				# Get Cosine Similarities
				A = m.embeddings[target_word_indices]
				B = m.embeddings[query_word_indices]
				print(np.shape(A), np.shape(B))
				cs_array[run_number] = np.matmul(A,B.T)
			except:
				continue

		cs_distribution = np.zeros((2, 1, query_word_num))
		cs_distribution[0] = np.mean(cs_array, axis = 0)
		cs_distribution[1] = np.std(cs_array, axis = 0)