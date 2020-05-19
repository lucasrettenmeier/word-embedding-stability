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
dist_folder = '/home/rettenls/data/experiments/wiki/analysis/distribution/'

languages = ['hi', 'en', 'fi', 'zh', 'cs', 'pl', 'pt']
models = ['word2vec', 'glove', 'fasttext']
data_type = 'shuffle'
model_type = 'skipgram'

total_run_number = 128

for language in languages:
	for model in models:
		# Get Folder
		if model == 'glove':
			folder = exp_folder + language + '/' + model + '/' + data_type
		else:
			folder = exp_folder + language + '/' + model + '/' + model_type + '/' + data_type

		analogy_results = list()
		for run_number in range(total_run_number):
			try:
				f = open(folder + '/run_{:04d}/eval.txt'.format(run_number))
				lines = f.readlines()
				analogy_results.append(float(lines[7][-8:-3].replace('(','')))
			except:
				pass

		try:
			print(language, model, data_type, len(analogy_results), np.mean(analogy_results), np.std(analogy_results))
		except:
			pass