#-------------------------------------------------------------------------------------------------------------------
# Packages & Settings
#-------------------------------------------------------------------------------------------------------------------

# General packages
import time
import sys
import os
import datetime
from glob import glob
import shutil

# Math and data structure packages
import numpy as np
from scipy import stats
import math
import matplotlib.pyplot as plt

# Writing Output
import pickle

exp_folder = '/home/rettenls/data/experiments/semeval/'

#-------------------------------------------------------------------------------------------------------------------
# Loading own Modules
#-------------------------------------------------------------------------------------------------------------------

import sys
sys.path.append("/home/rettenls/code/")

from lib.model 			import Model
from lib.trafo 			import Transformation
from lib.eval 			import print_nn_word, get_nn_list, get_cosine_similarity, get_pip_norm, get_ww_pip_norm
from lib.score 			import evaluate_analogy
from lib.operations 	import align, avg, align_list
from lib.util			import get_filename

#-------------------------------------------------------------------------------------------------------------------
# Checking the Coordination File
#-------------------------------------------------------------------------------------------------------------------

language = sys.argv[1]
corpora = ['corpus1', 'corpus2']

models = ['fasttext', 'word2vec', 'glove']
model_types = {'fasttext': ['skipgram'], 'word2vec' : ['skipgram'], 'glove' : [None]}

max_run_num = 16

results = dict()


m1 = Model('word2vec')
m1_folder = exp_folder + 'experiments/' + language + '/corpus' + str(1) + \
		'/word2vec/skipgram/shuffle/run_{:04d}'.format(0)
m1.load(m1_folder)

m2 = Model('word2vec')
m2_folder = exp_folder + 'experiments/' + language + '/corpus' + str(2) + \
		'/word2vec/skipgram/shuffle/run_{:04d}'.format(0)
m2.load(m2_folder)

# Find all words that occur more than 25 times in both corpora
eval_word_list = []
min_count = 25
for word_index_m1 in range(m1.voc_size):
	word = m1.words[word_index_m1]
	if word in m2.indices:
		word_index_m2 = m2.indices[word]
		if (m1.count[word_index_m1] > min_count and m2.count[word_index_m2] > min_count):
			eval_word_list.append(word)

for model in models:
	for model_type in model_types[model]:

		#-------------------------------------------------------------------------------------------------------
		#
		#	PART 1: Instability of Words Within the Two Corpora
		#
		#-------------------------------------------------------------------------------------------------------

		intra_epoch_variability = np.zeros((2,len(eval_word_list)))

		for corpus_num in [1,2]:

			for m1_num in range(max_run_num):

				# Load Model from Corpus 1
				if model == 'glove':
					m1_folder = exp_folder + 'experiments/' + language + '/corpus' + str(corpus_num) + \
						'/' + model + '/shuffle/run_{:04d}'.format(m1_num)
				else:
					m1_folder = exp_folder + 'experiments/' + language + '/corpus' + str(corpus_num) + \
						'/' + model + '/' + model_type + '/shuffle/run_{:04d}'.format(m1_num)
				m1.load(m1_folder)
				
				for m2_num in range(m1_num + 1, max_run_num):
					
					# Load Model from Corpus 2
					if model == 'glove':
						m2_folder = exp_folder + 'experiments/' + language + '/corpus' + str(corpus_num) + \
							'/' + model + '/shuffle/run_{:04d}'.format(m2_num)
					else:
						m2_folder = exp_folder + 'experiments/' + language + '/corpus' + str(corpus_num) + \
							'/' + model + '/' + model_type + '/shuffle/run_{:04d}'.format(m2_num)
					m2.load(m2_folder)

					# Align Models
					m1,m2,joint = align(m1,m2)
					
					# Get Indices of Eval Words
					eval_indices = [m1.indices[w] for w in eval_word_list]

					intra_epoch_variability[corpus_num - 1] += get_ww_pip_norm(model1 = m1, model2 = m2, 
							eval_indices = eval_indices, avg_indices = joint)

		intra_epoch_variability /= (max_run_num * (max_run_num - 1) / 2)

		#-------------------------------------------------------------------------------------------------------
		#
		#	PART 2: Instability of Words Between the Two Corpora
		#
		#-------------------------------------------------------------------------------------------------------

		inter_epoch_variability = np.zeros((len(eval_word_list)))

		for m1_num in range(max_run_num):

			# Load Model from Corpus 1
			if model == 'glove':
				m1_folder = exp_folder + 'experiments/' + language + '/corpus' + str(1) + \
					'/' + model + '/shuffle/run_{:04d}'.format(m1_num)
			else:
				m1_folder = exp_folder + 'experiments/' + language + '/corpus' + str(1) + \
					'/' + model + '/' + model_type + '/shuffle/run_{:04d}'.format(m1_num)
			m1.load(m1_folder)
			
			for m2_num in range(max_run_num):
				
				# Load Model from Corpus 2
				if model == 'glove':
					m2_folder = exp_folder + 'experiments/' + language + '/corpus' + str(2) + \
						'/' + model + '/shuffle/run_{:04d}'.format(m2_num)
				else:
					m2_folder = exp_folder + 'experiments/' + language + '/corpus' + str(2) + \
						'/' + model + '/' + model_type + '/shuffle/run_{:04d}'.format(m2_num)
				m2.load(m2_folder)

				# Align Models
				m1,m2,joint = align(m1,m2)
				
				# Get Indices of Eval Words
				eval_indices = [m1.indices[w] for w in eval_word_list]

				inter_epoch_variability += get_ww_pip_norm(model1 = m1, model2 = m2, 
						eval_indices = eval_indices, avg_indices = joint)

		inter_epoch_variability /= (max_run_num * max_run_num)
	results[model] = [intra_epoch_variability, inter_epoch_variability]

# Save Results and Word List
dist_folder = exp_folder + 'distributions/' + language +'/'	
pickle.dump(results, open(dist_folder + 'results.pickle', 'wb'))
pickle.dump(eval_word_list, open(dist_folder + 'words.pickle', 'wb'))
		