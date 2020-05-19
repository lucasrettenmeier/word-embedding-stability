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

date_format = '%Y-%m-%d_%H:%M:%S'

#-------------------------------------------------------------------------------------------------------------------
# Loading own Modules
#-------------------------------------------------------------------------------------------------------------------

sys.path.append('/home/rettenls/code')
#sys.path.append('/home/lucas/code')

from lib.model 			import Model
from lib.trafo 			import Transformation
from lib.eval 			import print_nn_word, get_nn_list, get_cosine_similarity, get_pip_norm, get_word_relatedness,get_common_vocab
from lib.score 			import evaluate_analogy
from lib.operations 	import align, avg, join, align_list
from lib.util			import get_filename
from scipy.stats		import spearmanr


#-------------------------------------------------------------------------------------------------------------------
# Experiments
#-------------------------------------------------------------------------------------------------------------------


text_folder = '/home/rettenls/data/texts/wiki/'
eval_folder = '/home/rettenls/data/evaluation/analogy/'
exp_folder = '/home/rettenls/data/experiments/wiki/'
dist_folder = '/home/rettenls/data/experiments/wiki/analysis/word-wise-instability/'
'''
text_folder = '/home/lucas/data/texts/wiki/'
eval_folder = '/home/lucas/data/evaluation/analogy/'
exp_folder = '/home/lucas/data/experiments/wiki/'
dist_folder = '/home/lucas/data/experiments/wiki/analysis/word-wise-instability/'
'''

languages = ['hi', 'fi', 'zh', 'cs', 'pl', 'pt', 'en']
languages = ['en']
models = ['fasttext', 'word2vec']#, 'glove']
data_types = ['shuffle', 'bootstrap']#, 'fixed']
model_type = 'skipgram'

total_run_number = 16

target_count = int(2.e3)
proxy_count = int(2.e4)

for language in languages:

	# Get Common Vocabulary over all Models & Data Types
	directory_list = list()
	for model in models:
		for data_type in data_types:

			# Get Folder
			if model == 'glove':
				folder = exp_folder + language + '/' + model + '/' + data_type
			else:
				folder = exp_folder + language + '/' + model + '/' + model_type + '/' + data_type

			# Get Common Vocabulary
			for run_number in range(total_run_number):
				directory_list.append(folder + '/run_{:04d}'.format(run_number))

	common_vocab = get_common_vocab(directory_list)
	print('Common vocabulary of all runs determined. Size:', len(common_vocab), 'words.')

	# Sample proxy words from the common vocabulary
	word_array = np.array(list(common_vocab))
	np.random.shuffle(word_array)
	word_array = word_array[:proxy_count]

	# Iterate over all Models & Data Types
	for model in models:
		for data_type in data_types:

			# Get Folder
			if model == 'glove':
				folder = exp_folder + language + '/' + model + '/' + data_type
			else:
				folder = exp_folder + language + '/' + model + '/' + model_type + '/' + data_type

			# Array for word counts
			word_counts = np.zeros((target_count))

			# Read Embeddings - Calculate PIP and store it
			pip = np.zeros((total_run_number, target_count, proxy_count))
			for run_number in range(total_run_number):

				# Load model
				m = Model(model)
				m.load(folder + '/run_{:04d}'.format(run_number))

				# Get word indices
				word_index_array = [m.indices[word] for word in word_array]

				# Get word counts
				word_counts += np.array([m.count[index] for index in word_index_array[:target_count]]) / m.total_count

				# Get word embeddings
				word_embedding_array = m.embeddings[word_index_array]
				#print(np.shape(word_embedding_array))
				
				# Calculate PIP Loss
				pip[run_number] = np.matmul(word_embedding_array[:target_count], word_embedding_array.T)

			# Calculate word wise reduced PIP Loss
			pip_loss_results = np.zeros((target_count, (total_run_number - 1) * total_run_number // 2))
			pair_index = 0
			for run_number_1 in range(total_run_number):
				for run_number_2 in range(run_number_1 + 1, total_run_number):
					pip_loss_results[:,pair_index] = np.sqrt(np.sum(np.square(pip[run_number_1] - pip[run_number_2]), axis = 1)) / (2 * np.sqrt(proxy_count))
					pair_index += 1			

			# Save
			np.savez(dist_folder + language + '_' + model + '_' + data_type, pip_loss_results, word_counts, np.array(word_array))