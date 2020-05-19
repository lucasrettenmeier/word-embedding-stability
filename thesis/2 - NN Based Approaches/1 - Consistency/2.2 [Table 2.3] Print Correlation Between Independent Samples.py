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
from scipy.stats import spearmanr

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
from lib.operations 	import align, avg, join, align_list, get_common_vocab
from lib.util			import get_filename


#-------------------------------------------------------------------------------------------------------------------
# Experiments
#-------------------------------------------------------------------------------------------------------------------

languages = ['hi', 'fi', 'zh', 'cs', 'pl', 'pt', 'en']
models = ['word2vec', 'glove', 'fasttext']

n_values = [2,5,10,25,50]

total = np.zeros((len(models), 2, len(n_values)))
cor = np.zeros((len(models), len(languages), 2, len(n_values)))

for lang_index in range(len(languages)):
	for model_index in range(len(models)):

		language = languages[lang_index]
		model = models[model_index]

		file_name = '/home/rettenls/code/eval/nn_metrics_evaluation/avg_size_control_' + language + '_' + model + '.pkl'
		try:
			results = pickle.load(open(file_name, 'rb'))
			for i in range(2):
				for j in range(len(n_values)):
					cor[model_index,lang_index,i,j] = results[i][j]
				total[model_index, i] += cor[model_index, lang_index, i]
		except:
			continue

total /= len(languages)
print(total)