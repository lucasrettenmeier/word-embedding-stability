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

languages = {'fasttext': ['en'], 'glove' : ['en', 'fi', 'pl'], 'word2vec': ['hi', 'cs', 'zh', 'pt', 'pl', 'fi', 'en']}

models = ['fasttext', 'glove', 'word2vec']
model_type = 'skipgram'
data_type = 'shuffle'

word_num = int(2 * 1.e4)
word_pair_num = word_num // 2

for model in models:
	for language in languages[model]:

		print(model, language)

		if model == 'glove':
			folder = exp_folder + language + '/' + model + '/' + data_type + '/'
		else:
			folder = exp_folder + language + '/' + model + '/' + model_type + '/' + data_type + '/'

		# Get folders
		runs = [run for run in os.listdir(folder) if (run[:15] == 'merge_nnz_0002_' and len(os.listdir(folder + run)) >= 5)][:64]

		# Get Common Vocabulary
		directory_list = [folder + run for run in runs]
		common_vocab = get_common_vocab(directory_list)

		# Sample proxy_count words from the common vocabulary
		word_array = np.array(list(common_vocab))
		np.random.shuffle(word_array)
		word_array = word_array[:word_num]
		
		# Store Cosine Similarity Values
		cos_sim = np.zeros((len(runs), word_pair_num))

		for run_number in range(len(runs)):
			run = folder + runs[run_number]

			m = Model(model)
			m.load(run)
			m.normalize()
			word_indices = [m.indices[word] for word in word_array]

			# Construct Two Arrays, each holds one embedding of each word pair
			index_arr_1 = np.array(word_indices[::2])
			index_arr_2 = np.array(word_indices[1::2])

			A = m.embeddings[index_arr_1]
			B = m.embeddings[index_arr_2]

			# Shortcut to get only diagonal elements of product A * B.T
			result = np.sum(A * B, axis = 1)
			cos_sim[run_number] = result
			
		np.save(file = dist_folder + 'merge_normalized_0002_' + language + '_' + model + '.npy', arr = cos_sim)