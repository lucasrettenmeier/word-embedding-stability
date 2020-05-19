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
data_folder = '/home/rettenls/data/experiments/wiki/analysis/overlap/N=1/'
ana_folder = '/home/rettenls/data/experiments/wiki/analysis/overlap/Fixed_Variance/'

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

languages = [ 'pt', 'zh', 'cs', 'pl', 'hi', 'fi', 'en']
languages = ['en']
models = ['fasttext', 'word2vec', 'glove']
model_types = {'word2vec': ['cbow'], 'fasttext': ['skipgram'], 'glove': [None]}
data_types = ['shuffle']

target_word_num = 1000
query_word_num = 100

for language in languages:
	for model in models:
		
		cs_distribution = np.load(file = data_folder + 'dist_' + language + '_' + model + '.npy')
		measured = np.load(file = data_folder + 'result_' + language + '_' + model + '.npy')[0]

		# Predict Overlap
		predicted = np.zeros(target_word_num)

		# Use Mean STD
		std = np.zeros(query_word_num) + np.mean(cs_distribution[1])

		for target_word_index in range(target_word_num):

			mean = cs_distribution[0][target_word_index]
			
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

			if(target_word_index % 10 == 0):
				print(str(target_word_index) + ' words completed...' , end = '\r')
	
		print('Language: ', language, '\nModel:', model, '\nSpearman:', spearmanr(measured, 
			predicted)[0] , '\nPearson:', pearsonr(measured, predicted)[0])

		np.save(file = ana_folder + 'result_' + language + '_' + model + '.npy', arr = [measured, predicted])