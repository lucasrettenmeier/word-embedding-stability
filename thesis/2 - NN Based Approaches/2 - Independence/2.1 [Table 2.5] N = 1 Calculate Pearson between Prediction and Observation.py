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
import math as ma
import random

# Plots, Fits, etc.
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.stats import spearmanr, pearsonr

# Writing Output
import pickle

text_folder = '/home/rettenls/data/texts/wiki/'
eval_folder = '/home/rettenls/data/evaluation/analogy/'
exp_folder = '/home/rettenls/data/experiments/wiki/'
ana_folder = '/home/rettenls/data/experiments/wiki/analysis/overlap/N=1/'

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
# Numerical Integration
#-------------------------------------------------------------------------------------------------------------------

def gaussian(x, mu, sigma):
    return (1 / (ma.sqrt(2*ma.pi) * sigma)) * ma.exp(- 0.5 * ma.pow((x - mu) / sigma, 2.))

def error_func(x, mu, sigma):
	return 0.5 * (ma.erf((x-mu)/(ma.sqrt(2) * sigma)) + 1)

def function_n1(x, i, params):
	result = gaussian(x,params[0][i], params[1][i])
	for j in range(len(params[0])):
		if j == i:
			continue
		result *= error_func(x, params[0][j], params[1][j])
	return result

def function_n2_helper(x,i,j,params):
	result = gaussian(x,params[0][i], params[1][i])
	for k in range(len(params[0])):
		if k == i or k == j:
			continue
		result *= error_func(x, params[0][k], params[1][k])
	return result

def function_n2(x,i,j,params):
	result = gaussian(x,params[0][i], params[1][i]) 
	return integrate.quad(lambda x: function_n1(x,rel_index,params),0,1)[0]

#-------------------------------------------------------------------------------------------------------------------
# Experiments
#-------------------------------------------------------------------------------------------------------------------

languages = ['fi', 'hi', 'zh', 'cs', 'pl', 'pt', 'en']
languages = ['en']
models = ['glove', 'word2vec', 'fasttext']
model_types = {'word2vec': ['cbow'], 'fasttext': ['skipgram'], 'glove': [None]}
data_types = ['shuffle']

target_word_num = 1000
query_word_num = 100

for language in languages:
	for model in models:

		model_type = model_types[model][0]
		data_type = data_types[0]

		if model == 'glove':
			folder = exp_folder + language + '/' + model + '/' + data_type + '/'
		else:
			folder = exp_folder + language + '/' + model + '/' + model_type + '/' + data_type + '/'

		# Get folders
		runs = [run for run in os.listdir(folder) if (run[:3] == 'run' and len(os.listdir(folder + run)) == 5)]

		# Get Common Vocabulary
		directory_list = [folder + run for run in runs]
		common_vocab = get_common_vocab(directory_list)

		# Sample proxy_count words from the common vocabulary
		word_array = common_vocab
		np.random.shuffle(word_array)
		word_array = word_array[:target_word_num]

		# Base Model
		m_base = Model(model)
		m_base.load(folder + runs[0])
		m_base = m_base.reduce_to_vocab(list(common_vocab))

		# Get Word Indices
		target_word_indices = [m_base.indices[word] for word in word_array]

		# Query Word Indices
		query_word_indices_base = get_nn_list(m_base, m_base, target_word_indices, query_word_num, False, True)[0]

		# Array for storing the COS-SIMs:
		cs_array = np.zeros((len(runs), target_word_num, query_word_num))

		# Load Models & Get NN Arrays
		for run_number in range(len(runs)):

			run = folder + runs[run_number]

			if run_number > 0:
				m = Model(model)
				m.load(run)
			else:
				m = m_base

			# Get Cosine Similarities

			target_word_indices = [m.indices[word] for word in word_array]
			for target_word_index in range(target_word_num):
				A = m.embeddings[np.array([target_word_indices[target_word_index]])]
				query_word_indices = [m.indices[m_base.words[index]] for index in query_word_indices_base[target_word_index]] 
				B = m.embeddings[query_word_indices]
				cs_array[run_number][target_word_index] = np.matmul(A,B.T)

		cs_distribution = np.zeros((2, target_word_num, query_word_num))
		cs_distribution[0] = np.mean(cs_array, axis = 0)
		cs_distribution[1] = np.std(cs_array, axis = 0)

		np.save(file = ana_folder + 'dist_' + language + '_' + model + '.npy', arr = cs_distribution)

		# Measure Overlap (p@1 = j@1)
		measured = np.zeros(target_word_num) 
		top = np.argmax(cs_array, axis = 2) 
		for i in range(len(runs)): 
			for j in range(i+1,len(runs)): 
				overlap = np.zeros(target_word_num) 
				overlap[top[i] == top[j]] = 1 
				measured += overlap 
		measured /= (len(runs) * (len(runs) - 1) / 2)

		# Predict Overlap
		predicted = np.zeros(target_word_num)

		for target_word_index in range(target_word_num):

			mean = cs_distribution[0][target_word_index]
			std = cs_distribution[1][target_word_index]

			# Step 1: Get Relevance
			relevance_threshold = 1.e-5
			relevance = np.zeros(query_word_num)
			
			# Get - on average - nearest neighbor
			nn_index = np.argmax(mean)
			relevance[nn_index] = 1

			for query_index in range(query_word_num):
				if query_index == nn_index:
					continue
				A = (mean[query_index] - mean[nn_index]) / ma.sqrt(2*(ma.pow(std[query_index], 2) \
					+ ma.pow(std[nn_index], 2)))
				relevance[query_index] = 0.5 * (1 + ma.erf(A))

			# Step 2: Get p_rank_1 for relevant words and predict overlap
			relevant_words = np.where(relevance > relevance_threshold)[0]
			params = cs_distribution[:,target_word_index,relevant_words]

			overlap = 0
			for rel_index in range(len(relevant_words)):
				p_rank_1 = integrate.quad(lambda x: function_n1(x,rel_index,params),0,1)[0]
				#print(rel_index, p_rank_1)
				overlap += ma.pow(p_rank_1,2)

			predicted[target_word_index] = overlap
		
		print('Language: ', language, '\nModel:', model, '\nSpearman:', spearmanr(measured, 
			predicted)[0] , '\nPearson:', pearsonr(measured, predicted)[0])

		np.save(file = ana_folder + 'result_' + language + '_' + model + '.npy', arr = [measured, predicted])