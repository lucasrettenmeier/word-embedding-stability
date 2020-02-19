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
data_types = ['bootstrap', 'shuffle']#, 'fixed']

max_run_num = 128
	
eval_size = int(2.e4)
max_eval_size = int(2.e3)
proxy_size = int(5.e4)

for language in languages:
	for model in models:
		for model_type in model_types:
			eval_words = 0
			avg_words = 0

			for data_type in data_types:

				if model == 'glove':
					folder = exp_folder + language + '/' + model + '/' + data_type
				else:
					folder = exp_folder + language + '/' + model + '/' + model_type + '/' + data_type

				
				instability_total = np.zeros(eval_size)
				counts_total = np.zeros(eval_size)

				if (data_type == 'bootstrap'):
					folder_list = list()
					for i in range(max_run_num):
						folder_list.append(folder + '/run_{:04d}'.format(i))

					if model == 'glove':
						folder_list.append(exp_folder + language + '/' + model + '/' + 'shuffle' + '/run_{:04d}'.format(0))
					else:
						folder_list.append(exp_folder + language + '/' + model + '/' + model_type + '/' + 'shuffle' + '/run_{:04d}'.format(0))

					common_vocab = get_common_vocab(folder_list)

					# Eval Indices	
					eval_indices = np.arange(len(common_vocab))
					np.random.shuffle(eval_indices)
					eval_indices = eval_indices[:eval_size]
					eval_words = common_vocab[eval_indices]

					# Average Indices
					avg_indices = np.arange(len(common_vocab))
					np.random.shuffle(avg_indices)
					avg_indices = avg_indices[:proxy_size]
					avg_words = common_vocab[avg_indices]

				for i in range(int(max_run_num / 2)):
					model_list = list()
					for j in range(2):
						m = Model(model)
						m.load(folder + '/run_{:04d}'.format(i * 2 + j))
						model_list.append(m)

					model_list[0], model_list[1], joint = align(model_list[0], model_list[1])

					alignment_correct = True
					for j in range(10):
						index = random.randint(0, len(common_vocab) - 1)
						for l in range(1, len(model_list)):
							if (model_list[0].words[index] != model_list[l].words[index]):
								alignment_correct = False 

					print("Vocabularies aligned:" , alignment_correct, '\n')

					# Determine the respective indices:
					for i in range(eval_size):
						eval_indices[i] = model_list[0].indices[eval_words[i]]

					for i in range(proxy_size):
						avg_indices[i] = model_list[0].indices[avg_words[i]]

					eval_steps = eval_size // max_eval_size

					for eval_step in range(eval_steps):

						lower = max_eval_size * eval_step
						upper = min(max_eval_size * (eval_step + 1), eval_size)

						eval_step_indices = eval_indices[lower:upper]
					
						pip_matrix = np.matmul(model_list[0].embeddings[eval_step_indices], model_list[0].embeddings[avg_indices].T) - \
										np.matmul(model_list[1].embeddings[eval_step_indices], model_list[1].embeddings[avg_indices].T) 
						instability = np.sqrt(np.sum(np.square(pip_matrix), axis = 1)) / math.sqrt(len(avg_indices))
						
						instability_total[lower:upper] += instability
						counts_total[lower:upper] += model_list[0].count[eval_step_indices] + model_list[1].count[eval_step_indices]

						print(upper, "of", eval_size, "words evaluated...", end = '\r')

					print('')

				instability_total /= int(max_run_num / 2)
				counts_total /= max_run_num

				np.savez('/home/rettenls/data/analysis/instability_over_frequency_' + model + '_' + language + '_' + data_type, x = counts_total, y = instability_total)