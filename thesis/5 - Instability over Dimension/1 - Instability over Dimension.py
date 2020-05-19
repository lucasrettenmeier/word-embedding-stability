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
from lib.eval 			import print_nn_word, get_nn_list, get_cosine_similarity, get_pip_norm, get_word_relatedness, get_common_vocab
from lib.score 			import evaluate_analogy
from lib.operations 	import align, avg, join, align_list
from lib.util			import get_filename
from scipy.stats		import spearmanr


#-------------------------------------------------------------------------------------------------------------------
# Experiments
#-------------------------------------------------------------------------------------------------------------------

models = ['word2vec']#, 'fasttext', 'glove']
dimension_number = 1000
vocabulary_sizes = [100,200,500,1000,2000,5000,10000,20000,50000, 'full']
large_vocabulary_sizes = [50000, 'full']

voc_folder = '/home/rettenls/data/texts/other/text8/vocabs/'
text_folder = '/home/rettenls/data/texts/other/text8/reduced/'
exp_folder = '/home/rettenls/data/experiments/other/vocabulary_sizes/'
dist_folder = '/home/rettenls/data/experiments/wiki/analysis/dimension/'

max_model_num = 8

results = list()

for model in models:
	for voc_size in vocabulary_sizes:
		for model_num in range(max_model_num):

			# Only Train if it does not exist
			if not os.path.isdir(exp_folder + '/' + model + '/' + str(voc_size) + '/fixed' + str(model_num)):
				m = Model(model)
				if (voc_size == 'full'):
					m.train(train_data_file = text_folder, dim_num = dimension_number, subsampling_rate = 0,
						epochs = 10)
				else:
					m.train(train_data_file = text_folder + 'voc_size_' + str(voc_size) + '.txt',
						dim_num = dimension_number,
						vocab_file = voc_folder + 'voc_size_' + str(voc_size),
						subsampling_rate = 0,
						epochs = 10)
				m.save(exp_folder + '/' + model + '/' + str(voc_size) + '/fixed' + str(model_num))

		# Get Common Vocabulary
		directory_list = list()
		for model_num in range(max_model_num):
			directory_list.append(exp_folder + '/' + model + '/' + str(voc_size) + '/fixed' + str(model_num))
		common_vocab = get_common_vocab(directory_list)

		print('Common vocabulary of all runs determined. Size:', len(common_vocab), 'words.')


		# Sample proxy_count words from the common vocabulary
		word_array = np.array(list(common_vocab))
		np.random.shuffle(word_array)
		if(voc_size in large_vocabulary_sizes):
			word_array = word_array[:20000]
			
		# Read Embeddings - Calculate PIP and store it
		pip = np.zeros((max_model_num, len(word_array), len(word_array)))
		for model_num in range(max_model_num):
			m = Model(model)
			m.load(exp_folder + '/' + model + '/' + str(voc_size) + '/fixed' + str(model_num))

			word_index_array = [m.indices[word] for word in word_array]
			word_embedding_array = m.embeddings[word_index_array]
			print(np.shape(word_embedding_array))
			pip[model_num] = np.matmul(word_embedding_array, word_embedding_array.T)

		# Calculate reduced PIP Loss
		pip_loss_results = list()
		for run_number_1 in range(max_model_num):
			for run_number_2 in range(run_number_1 + 1, max_model_num):
				pip_loss = np.sqrt(np.sum(np.square(pip[run_number_1] - pip[run_number_2])))
				pip_loss /= (2 * len(word_array))
				pip_loss_results.append(pip_loss)
				#print(run_number_1, run_number_2, pip_loss)

		print(model, voc_size, np.mean(pip_loss_results), np.std(pip_loss_results))
		results.append([np.mean(pip_loss_results), np.std(pip_loss_results)])

pickle.dump(results, open(dist_folder + 'data.pkl', 'wb'))