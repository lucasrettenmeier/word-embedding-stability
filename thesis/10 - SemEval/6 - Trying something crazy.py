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
from scipy import stats
from scipy.optimize import curve_fit
from pylab import *

import numpy as np
import math
import matplotlib.pyplot as plt

import pickle as pkl

from scipy.stats import spearmanr

data_folder = '/home/rettenls/data/experiments/semeval/texts/'
exp_folder = '/home/rettenls/data/experiments/semeval/experiments/'

#-------------------------------------------------------------------------------------------------------------------
# Loading own Modules
#-------------------------------------------------------------------------------------------------------------------

import sys
sys.path.append("/home/rettenls/code/")

from lib.model 			import Model
from lib.trafo 			import Transformation
from lib.eval 			import print_nn_word, get_nn_list, get_cosine_similarity, get_ww_pip_norm
from lib.score 			import evaluate_analogy
from lib.operations 	import align, avg
from lib.util			import get_filename
from lib.prepare	 	import bootstrap_corpus, shuffle_corpus, concatenate_files 

#-------------------------------------------------------------------------------------------------------------------
# Experiments
#-------------------------------------------------------------------------------------------------------------------

languages = ['english', 'german', 'latin', 'swedish']
models = ['fasttext', 'word2vec', 'glove']
model_types = {'word2vec': ['skipgram'], 'fasttext': ['skipgram'], 'glove': [None]}
corpora = ['corpus1', 'corpus2']


models = ['fasttext', 'word2vec']

sizes = [32]
max_run_num = 32

res = dict()
for model in models:
	for language in languages:
		for model_type in model_types[model]:
			for size in sizes:

				specific_results = list()

				
				# Task 1
				ans_folder = '/home/rettenls/data/experiments/semeval/golden_data/answer/task1/'
				answer_file_name = ans_folder + language + '.txt'
				answer_file = open(answer_file_name, 'r').readlines()
				answer_words = list()
				answer_bin = list()
				for line in answer_file:
					data = line.split('\t')
					answer_words.append(data[0])
					answer_bin.append(int(data[1][:-1]))
				answer_bin = np.array(answer_bin)

				# Task 2
				ans_folder = '/home/rettenls/data/experiments/semeval/golden_data/answer/task2/'
				answer_file_name = ans_folder + language + '.txt'
				answer_file = open(answer_file_name, 'r').readlines()
				answer_scores = list()
				for line in answer_file:
					data = line.split('\t')
					answer_scores.append(float(data[1][:-1]))

				# SHUFFLE

				data_type = 'shuffle'
				if model_type is None:
					folder1 = exp_folder + language + '/' + corpora[0] + '/' + model + '/' + data_type
				else:
					folder1 = exp_folder + language + '/' + corpora[0] + '/' + model + '/' + model_type + '/' + data_type

				if model_type is None:
					folder2 = exp_folder + language + '/' + corpora[1] + '/' + model + '/' + data_type
				else:
					folder2 = exp_folder + language + '/' + corpora[1] + '/' + model + '/' + model_type + '/' + data_type
		
				run_folder1 = folder1 + '/merge_{:04d}_run_{:04d}'.format(size, 0)
				run_folder2 = folder2 + '/merge_{:04d}_run_{:04d}'.format(size, 0)

				m1s = Model(model)
				m1s.load(run_folder1)

				m2s = Model(model)
				m2s.load(run_folder2)

				# BOOTSTRAP

				data_type = 'bootstrap'

				if model_type is None:
					folder1 = exp_folder + language + '/' + corpora[0] + '/' + model + '/' + data_type
				else:
					folder1 = exp_folder + language + '/' + corpora[0] + '/' + model + '/' + model_type + '/' + data_type

				if model_type is None:
					folder2 = exp_folder + language + '/' + corpora[1] + '/' + model + '/' + data_type
				else:
					folder2 = exp_folder + language + '/' + corpora[1] + '/' + model + '/' + model_type + '/' + data_type

				run_folder1 = folder1 + '/merge_{:04d}_run_{:04d}'.format(size, 0)
				run_folder2 = folder2 + '/merge_{:04d}_run_{:04d}'.format(size, 0)

				m1b = Model(model)
				m1b.load(run_folder1)

				m2b = Model(model)
				m2b.load(run_folder2)

				m1s,m1b,joint = align(m1s,m1b)
				t = Transformation('orthogonal', train_at_init = True, model1 = m1s, model2 = m1b, joint = joint)
				m1 = avg(t.apply_to(m1s),m1b)	

				m2s,m2b,joint = align(m2s,m2b)
				t = Transformation('orthogonal', train_at_init = True, model1 = m2s, model2 = m2b, joint = joint)
				m2 = avg(t.apply_to(m2s),m2b)	


				m1.normalize()
				m2.normalize()

				m1,m2,joint = align(m1,m2)
				eval_indices = [m1.indices[w] for w in answer_words]
				t = Transformation('orthogonal', train_at_init = True, model1 = m1, model2 = m2, joint = joint)						
				
				disp = 1 - get_cosine_similarity(t.apply_to(m1), m2, word_indices = joint)
				treshold = np.mean(disp) + 0.5 * np.std(disp)

				binary = np.array([int(disp[i] > treshold) for i in eval_indices])
				ranking = np.array([disp[i] for i in eval_indices])
	
				print(model + '_' + language + '_' + data_type + '_' + str(size), np.mean(binary == answer_bin), spearmanr(answer_scores, ranking)[0])
				res[model + '_' + language + '_' + data_type + '_' + str(size)] = (np.mean(binary == answer_bin), spearmanr(answer_scores, ranking)[0])
