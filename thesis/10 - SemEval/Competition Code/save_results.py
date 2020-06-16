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

from scipy.stats import spearmanr

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

def open_file(exp_folder, submission, task, language):

	folder = exp_folder + 'mixed_answers/' + submission + '/answer/' + task

	if not os.path.isdir(folder):
		os.makedirs(folder)

	file_name = folder + '/' + language
	return (open(file_name, 'w'))

def get_displacement(dist):
	displacement = dist# / (np.sqrt(np.square(dist[0][0]) + np.square(dist[0][1])))
	displacement /= np.mean(displacement)
	return (displacement)

def gauss(x,mu,sigma,A):
	return A*exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

def fit_double_gauss(displacement, show = False):
	data = displacement
	y,x,_= plt.hist(data,100,alpha=.3,label='data')

	x=(x[1:]+x[:-1])/2 # for len(x)==len(y)

	expected=(	np.mean(displacement),
				np.std(displacement),
				max(y),
				np.mean(displacement) + np.std(displacement),
				np.std(displacement),
				max(y) / 5)

	params,cov=curve_fit(bimodal,x,y,expected)
	sigma=sqrt(diag(cov))
	plt.plot(x,gauss(x,*params[:3]),color='red',lw=3,label='model_1')
	plt.plot(x,gauss(x,*params[3:]),color='blue',lw=3,label='model_2')
	legend()
	if show:
		plt.show()
	return params   

languages = ['english', 'german', 'latin', 'swedish']
corpora = ['corpus1', 'corpus2']

submissions = ['fasttext_glove_word2vec', 'fasttext_glove', 'word2vec_glove', 
	'fasttext_word2vec', 'fasttext', 'glove', 'word2vec']

max_run_num = 16

for language in languages:

	# Open Task File
	task_file_name = exp_folder + 'tasks/targets/' + language + '/targets.txt'
	task_file = open(task_file_name, 'r')
	task_file_lines = task_file.readlines()
	task_file.close()

	# Read Eval Words from File
	eval_words = []
	for task_file_line in task_file_lines:
		word = task_file_line.split('\n')[0]
		eval_words.append(word)

	# Get Word List and Distributions from Disk
	dist_folder = exp_folder + 'mixed_distributions/' + language + '/'	
	
	distributions 	= pickle.load(open(dist_folder + 'results.pickle', 'rb'))
	word_list 		= pickle.load(open(dist_folder + 'words.pickle', 'rb'))
			
	for submission in submissions:
		models = submission.split('_')

		#if(len(models) == 2):
		#	continue

		semantic_displacement = None

		for model in models:
			if semantic_displacement is None:
				semantic_displacement = get_displacement(distributions[model])
			else:
				semantic_displacement += get_displacement(distributions[model])

		fit_params = fit_double_gauss(semantic_displacement, True)
		if (fit_params[0] < fit_params[3]):
			mean = fit_params[0]
			std = fit_params[1]
		else:
			mean = fit_params[3]
			std = fit_params[4]

		# Calculate Results
		task1_result = list()
		task2_result = list()
		for word in eval_words:
			word_index = word_list.index(word)
			word_displacement = semantic_displacement[word_index]
			
			# Task 1
			if (word_displacement > mean + std):
				task1_result.append(1)
			else:
				task1_result.append(0)

			# Task 2
			task2_result.append(word_displacement)

		# Open Files
		task1_file = open_file(exp_folder, submission, 'task1', language + '.txt')
		task2_file = open_file(exp_folder, submission, 'task2', language + '.txt')

		# Write to File
		i = 0
		for word in eval_words:
			task1_res_line = word + '\t' + str(task1_result[i]) + '\n'
			task1_file.write(task1_res_line)

			task2_res_line = word + '\t' + str(task2_result[i]) + '\n'
			task2_file.write(task2_res_line)

			i += 1

		task1_file.close()
		task2_file.close()			

		print('Completed Evaluation.')
		print('Language:', language)
		print('Models:', models)
		print('Evaluation:', len([x for x in task1_result if x == 1]), 'of', len(task1_result), 'have changed in meaning.')
		print('')