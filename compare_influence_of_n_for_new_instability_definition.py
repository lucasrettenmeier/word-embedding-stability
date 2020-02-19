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

from lib.model 			import Model
from lib.trafo 			import Transformation
from lib.eval 			import print_nn_word, get_nn_list, get_cosine_similarity, get_pip_norm, get_word_relatedness
from lib.score 			import evaluate_analogy
from lib.operations 	import align, avg, join, align_list, get_common_vocab
from lib.util			import get_filename


#-------------------------------------------------------------------------------------------------------------------
# Experiments
#-------------------------------------------------------------------------------------------------------------------

languages = ['fi', 'cs', 'hi', 'zh', 'pt', 'pl']#, 'it', 'es', 'fr', 'de', 'en']
models = ['fasttext', 'glove', 'word2vec']
model_types = ['skipgram']
data_types = ['shuffle']#, 'fixed']

max_run_num = 10
	
eval_size = int(1.e2)

avg_sizes = [int(1.e3),int(1.e4),int(1.e5),int(1.e6)]
avg_max_size = max(avg_sizes)

for language in languages:
	for model in models:
		for model_type in model_types:
			for data_type in data_types:

				if model == 'glove':
					folder = exp_folder + language + '/' + model + '/' + data_type
				else:
					folder = exp_folder + language + '/' + model + '/' + model_type + '/' + data_type

				m_base = Model(model)
				m_base.load(folder + '/run_{:04d}'.format(0))

				# Eval Indices	
				eval_indices = np.arange(m_base.voc_size)
				np.random.shuffle(eval_indices)
				eval_indices = eval_indices[:eval_size]

				model_list = list()
				model_list.append(m_base)
				i = 0 
				
				for i in range(1, max_run_num):
					m = Model(model)
					m.load(folder + '/run_{:04d}'.format(i))
					model_list.append(m)
				
				model_list = align_list(model_list)

				instability_matrix = np.zeros((eval_size, len(avg_sizes)))

				i = 0
				for avg_size in avg_sizes:

					# AVG Indices	
					avg_indices = np.arange(m_base.voc_size)
					np.random.shuffle(avg_indices)
					avg_indices = avg_indices[:avg_size]

					#Get the stability
					for m1 in range(max_run_num):
						for m2 in range(m1+1, max_run_num):

							pip_matrix = np.matmul(model_list[m1].embeddings[eval_indices], model_list[m1].embeddings[avg_indices].T) - \
											np.matmul(model_list[m2].embeddings[eval_indices], model_list[m2].embeddings[avg_indices].T)

							instability_matrix[:,i] += np.sqrt(np.sum(np.square(pip_matrix), axis = 1)) / math.sqrt(len(avg_indices))

					i += 1
				
				for i in range(len(avg_sizes)):
					for j in range(i+1,len(avg_sizes)):
						print(i,j,stats.spearmanr(instability_matrix[:,i], instability_matrix[:,j]))
				break
			break
		break
	break