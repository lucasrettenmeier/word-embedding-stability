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
dist_folder = '/home/rettenls/data/experiments/wiki/analysis/distribution/'

languages = ['hi', 'fi', 'zh', 'cs', 'pl', 'pt', 'en']
languages = ['pt']
models = ['fasttext', 'glove', 'word2vec']
models = ['glove']
data_types = ['fixed']
model_type = 'skipgram'

total_run_number = 4

proxy_count = int(1.8e4)

results = list()
for language in languages:
	lang_results = list()
	for model in models:
		model_results = list()
		for data_type in data_types:

			# Get Folder
			if model == 'glove':
				folder = exp_folder + language + '/' + model + '/' + data_type
			else:
				folder = exp_folder + language + '/' + model + '/' + model_type + '/' + data_type

			# Get Common Vocabulary
			directory_list = list()
			for run_number in range(total_run_number):
				directory_list.append(folder + '/run_{:04d}'.format(1 + run_number))
			common_vocab = get_common_vocab(directory_list)

			print('Common vocabulary of all runs determined. Size:', len(common_vocab), 'words.')

			# Sample proxy_count words from the common vocabulary
			word_array = np.array(list(common_vocab))
			np.random.shuffle(word_array)
			word_array = word_array[:proxy_count]

			# Read Embeddings - Calculate PIP and store it
			pip = np.zeros((total_run_number, proxy_count, proxy_count))
			for run_number in range(total_run_number):
				print(run_number)
				m = Model(model)
				m.load(folder + '/run_{:04d}'.format(1 + run_number))

				word_index_array = [m.indices[word] for word in word_array]
				word_embedding_array = m.embeddings[word_index_array]
				#print(np.shape(word_embedding_array))
				pip[run_number] = np.matmul(word_embedding_array, word_embedding_array.T)

			# Calculate reduced PIP Loss
			pip_loss_results = list()
			for run_number_1 in range(total_run_number):
				for run_number_2 in range(run_number_1 + 1, total_run_number):
					pip_loss = np.sqrt(np.sum(np.square(pip[run_number_1] - pip[run_number_2])))
					pip_loss /= (2 * proxy_count)
					pip_loss_results.append(pip_loss)
					print(run_number_1, run_number_2, pip_loss)

			print(language, model, data_type, np.mean(pip_loss_results), np.std(pip_loss_results))

			model_results.append([np.mean(pip_loss_results), np.std(pip_loss_results)])
		lang_results.append(model_results)
	results.append(lang_results)
