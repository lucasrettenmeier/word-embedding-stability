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

from lib.model 			import Model
from lib.trafo 			import Transformation
from lib.eval 			import print_nn_word, get_nn_list, get_cosine_similarity, get_pip_norm, get_word_relatedness
from lib.score 			import evaluate_analogy
from lib.operations 	import align, avg, join
from lib.util			import get_filename


#-------------------------------------------------------------------------------------------------------------------
# Experiments
#-------------------------------------------------------------------------------------------------------------------

languages = ['fi', 'hi', 'cs', 'zh', 'pt', 'pl', 'it']#, 'es', 'fr', 'de', 'en']
models = ['glove', 'fasttext', 'word2vec']
model_types = ['skipgram']
data_types = ['shuffle']#, 'bootstrap', 'fixed']

max_run_num = 128

for language in languages:
	for model in models:
		for model_type in model_types:
			for data_type in data_types:

				if model == 'glove':
					folder = exp_folder + language + '/' + model + '/' + data_type
				else:
					folder = exp_folder + language + '/' + model + '/' + model_type + '/' + data_type

				model_set = [x for x in os.listdir(folder) if (x[:4] == 'run_'  and len(os.listdir(folder + '/' + x)) >= 3)]
				number_of_models 	= len(model_set)
				number_of_words 	= int(1.e3)
				number_of_pairs 	= int((number_of_words ** 2 - number_of_words) / 2)

				print(folder, number_of_models)

				if (model == 'glove'):

					# Load model
					m = Model(model)
					m.load(folder + '/run_0000')

					# Get random indices 
					word_indices = np.arange(m.voc_size)
					np.random.shuffle(word_indices)

					# Write words to list 
					word_list = list()
					for i in range(number_of_words):
						word_list.append(m.words[word_indices[i]])
					

				results = np.zeros((number_of_models,number_of_pairs))

				model_num = 0
				for model_name in model_set:
					m = Model(model)
					m.load(folder + '/' + model_name)

					k = 0
					for i in range(len(word_list)):
						for j in range(i + 1, len(word_list)):
							word1 = word_list[i]
							word2 = word_list[j]
							results[model_num][k] = get_word_relatedness(m, word1, word2)
							k += 1

					model_num += 1

				np.save(dist_folder + '/' + language + '_' + model + '_128.npy', results)

				word_arr = np.array(word_list)
				np.save(dist_folder + '/' + language + '_' + model + '_128_words.npy', word_arr)