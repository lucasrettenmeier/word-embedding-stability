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

languages = ['cs', 'fi', 'zh', 'hi', 'pl', 'pt']
models = ['fasttext', 'word2vec']

model_type = 'skipgram'
data_type = 'shuffle'

epochs = [5,10,20,40]
ns_sizes = [5,10,20]

total_run_number = 4
proxy_count = int(2.e4)


for language in languages:
	for model in models:

		# Get Folder
		if model == 'glove':
			folder = exp_folder + language + '/' + model + '/' + data_type
		else:
			folder = exp_folder + language + '/' + model + '/' + model_type + '/' + data_type

		# Get Common Vocabulary
		directory_list = list()
		directory_list.append(folder + '/run_{:04d}'.format(0))
		common_vocab = get_common_vocab(directory_list)

		# Sample proxy_count words from the common vocabulary
		word_array = np.array(list(common_vocab))
		np.random.shuffle(word_array)
		word_array = word_array[:proxy_count]

		# Get Base Models
		base_pip = np.zeros((total_run_number, proxy_count, proxy_count))
		for run_number in range(total_run_number):
			m = Model(model)
			m.load(folder + '/run_{:04d}'.format(total_run_number + run_number))
			word_index_array = [m.indices[word] for word in word_array]
			word_embedding_array = m.embeddings[word_index_array]
			base_pip[run_number] = np.matmul(word_embedding_array, word_embedding_array.T)
			
		for epoch in epochs:
			for ns in ns_sizes:

				# Read Embeddings - Calculate PIP and store it
				test_pip = np.zeros((total_run_number, proxy_count, proxy_count))
				for run_number in range(total_run_number):
					m = Model(model)
					if epoch == 5 and ns == 5:
						m.load(folder + '/run_{:04d}'.format(run_number))
					else:
						m.load(folder + '/ep_{:04d}_ns_{:04d}_run_{:04d}'.format(epoch, ns, run_number))
					word_index_array = [m.indices[word] for word in word_array]
					word_embedding_array = m.embeddings[word_index_array]
					test_pip[run_number] = np.matmul(word_embedding_array, word_embedding_array.T)

				# Calculate reduced PIP Loss
				pip_loss_results = list()
				for base_run_number in range(total_run_number):
					for test_run_number in range(total_run_number):
						pip_loss = np.sqrt(np.sum(np.square(base_pip[base_run_number] - test_pip[test_run_number])))
						pip_loss /= (2 * proxy_count)
						pip_loss_results.append(pip_loss)
					
				print(language, model, epoch, ns, len(pip_loss_results), np.mean(pip_loss_results))
