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

languages = ['hi', 'fi', 'zh', 'cs', 'pl', 'pt']
languages = ['en']

models = ['fasttext', 'glove', 'word2vec']

model_type = 'skipgram'
data_types = ['shuffle']

for language in languages:
	for model in models:
		for data_type in data_types:

			# Get Folder
			if model == 'glove':
				folder = exp_folder + language + '/' + model + '/' + data_type
			else:
				folder = exp_folder + language + '/' + model + '/' + model_type + '/' + data_type
		
			
			run_folders = [run for run in os.listdir(folder) if run[:5] == 'run_0'][:128]
			print(language, model, len(run_folders))

			base_results = list()
			for run in run_folders:
				try:
					f = open(folder + '/' + run + '/eval.txt')
					lines = f.readlines()
					base_results.append(float(lines[7][-8:-3].replace('(','')))
				except:
					pass

			print(language, model, len(base_results), np.mean(base_results), np.std(base_results))


			merge_run_folders = [run for run in os.listdir(folder) if run[:14] == 'merge_nnz_0008'][:32]
			print(language, model, len(merge_run_folders))

			merge_results = list()
			for run in merge_run_folders:
				try:
					f = open(folder + '/' + run + '/eval.txt')
					lines = f.readlines()
					merge_results.append(float(lines[7][-8:-3].replace('(','')))
				except:
					pass

			print(language, model, len(merge_results), np.mean(merge_results), np.std(merge_results))

