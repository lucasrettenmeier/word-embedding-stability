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
# Experiments
#-------------------------------------------------------------------------------------------------------------------

languages = ['hi', 'fi', 'zh', 'cs', 'pl', 'pt', 'en']
models = ['fasttext', 'word2vec', 'glove']
model_types = {'word2vec': ['cbow'], 'fasttext': ['skipgram'], 'glove': [None]}
data_types = ['shuffle']

total_run_num = 16
neighbor_num = 10

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

		# Random Target Word Indices
		#target_word_num = 1000
		#target_word_indices = np.arange(m_base.voc_size) # Sample Target Words from the 200,000 Most Frequent Words
		#np.random.shuffle(target_word_indices)
		#target_word_indices = target_word_indices[:target_word_num]

		# Target Words From Eval File
		eval_file_name = eval_folder + 'questions-words-' + language + '.txt'
		eval_file = open(eval_file_name, 'r')
		eval_lines = eval_file.readlines()
		eval_words = set()
		for line in eval_lines:
			if line[:2] == ': ':
				continue
			line = line.replace('\n', '')
			words = line.split(' ')
			for word in words: 
				eval_words.add(word)
		target_word_indices = [m_base.indices[word] for word in eval_words if word in m_base.indices]
		target_word_num = len(target_word_indices)

		# Query Word Indices
		nn_array = np.zeros((total_run_num, target_word_num, neighbor_num))

		# Model List
		model_list = list()

		# Load Models & Get NN Arrays
		for run_number in range(total_run_num):

			if run_number > 0:
				m = Model(model)
				m.load(folder + '/run_{:04d}'.format(run_number))
				_, m, __ = align(m_base,m)
				
			else:
				m = m_base

			model_list.append(m)
			nn_array[run_number] = get_nn_list(m, m, target_word_indices, neighbor_num, False, False)


		# Get Scores on Analogy Tasks for Each Model
		scores = list() 
		for i in range(total_run_num): 
			score_file_name = folder + '/run_{:04d}/eval.txt'.format(i) 
			score_file = open(score_file_name, 'r') 
			score_text = score_file.readlines()[7][55:60]
			scores.append(float(score_text.replace('%','')))

		# Measure Overlap & PIP Loss
		overlap = np.zeros( ( int(total_run_num * (total_run_num - 1) / 2) , target_word_num) )
		pip_loss = np.zeros( int(total_run_num * (total_run_num - 1) / 2) )
		score_diffs = np.zeros( int(total_run_num * (total_run_num - 1) / 2) )

		pair_index = 0
		for i in range(total_run_num): 
			for j in range(i+1,total_run_num):

				# Overlap
				for target_word_index in range(target_word_num):
					overlap[pair_index][target_word_index] = len(np.intersect1d(nn_array[i][target_word_index], nn_array[j][target_word_index])) / neighbor_num
					
				# PIP Loss
				pip_loss[pair_index] = get_pip_norm(model1 = model_list[i], model2= model_list[j], reduced = True, get_proxy = True)

				# SCore
				score_diffs[pair_index] = abs(scores[i] - scores[j])

				# Next Pair
				pair_index += 1

		overlap_sum = np.sum(overlap, axis = 1)

		
		print(model, language)
		print(spearmanr(overlap_sum, score_diffs))
		print(spearmanr(pip_loss, score_diffs))
	'''
	
	print('Language: ', language, '\nModel:', model, '\nSpearman:', spearmanr(measured, 
		predicted)[0] , '\nPearson:', pearsonr(measured, predicted)[0])

	np.save(file = ana_folder + 'result_' + language + '_' + model + '.npy', arr = [measured, predicted])

	'''