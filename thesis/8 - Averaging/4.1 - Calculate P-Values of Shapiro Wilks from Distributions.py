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
from lib.eval 			import print_nn_word, get_nn_list, get_cosine_similarity, get_pip_norm, get_word_relatedness
from lib.score 			import evaluate_analogy
from lib.operations 	import align, avg, join, align_list
from lib.util			import get_filename
from scipy.stats		import spearmanr, shapiro


#-------------------------------------------------------------------------------------------------------------------
# Experiments
#-------------------------------------------------------------------------------------------------------------------

languages = ['fi', 'hi', 'cs', 'zh', 'pt', 'pl', 'en']
models = ['word2vec', 'glove', 'fasttext']

languages = ['fi']
models = ['fasttext']

independence_p_values = np.zeros((len(models), len(languages), 5000))
gauss_p_values = np.zeros((len(models), len(languages), 10000))

for model_index in range(len(models)):
	model = models[model_index]
	for lang_index in range(len(languages)):
		try:
			language = languages[lang_index]

			# Get Cos-Sim Values
			cos_sim = np.load(file = dist_folder + 'merge_0002_' + language + '_' + model + '.npy')
			print(language, model, np.shape(cos_sim))
			word_pair_num = np.shape(cos_sim)[1]

			# Check Independence:
			p_values_ind = list()
			for i in range(word_pair_num // 2):
				p = spearmanr(cos_sim[:,i * 2], cos_sim[:,i * 2 + 1])[1]
				p_values_ind.append(p)
			independence_p_values[model_index][lang_index] = p_values_ind


			# Check Normal_Distribution:
			p_values_norm = list()
			for i in range(word_pair_num):
				p = shapiro(cos_sim[:,i])[1]
				p_values_norm.append(p)
			gauss_p_values[model_index][lang_index] = p_values_norm

			print(language, model, 'successful')

		except:
			print(language, model, 'skipped')

np.save(file = dist_folder + 'merge_0002_independence_p_values.npy', arr = independence_p_values)
np.save(file = dist_folder + 'merge_0002_gauss_p_values.npy', arr = gauss_p_values)