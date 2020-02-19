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

language = 'fi'
model = 'fasttext'
model_type = 'skipgram'
data_type = 'shuffle'

max_run_num = 10
	
eval_size = int(100)

nn_sizes = [5,10,50,100]
nn_max_size = 400

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

overlap = np.zeros((2,eval_size, len(nn_sizes)))
jaccard = np.zeros((2,eval_size, len(nn_sizes)))
slope = np.zeros((2,eval_size, len(nn_sizes)))

i = 0 

for full_run in range(2):
	model_list = list()

	for i in range(full_run * max_run_num, (full_run + 1) * max_run_num):
		m = Model(model)
		m.load(folder + '/run_{:04d}'.format(i))
		_, m, __ = align(m_base, m)
		model_list.append(m)

	# Array for storing the NNs:
	nn_array = np.zeros((max_run_num, eval_size, nn_max_size))
	cos_array = np.zeros((max_run_num, eval_size, nn_max_size))

	# Get the NNs
	for j in range(max_run_num):
		return_val = get_nn_list(model_list[j], model_list[j], eval_indices, nn_max_size, False, True)
		nn_array[j] = return_val[0]
		cos_array[j] = return_val[1]

	#Analyse the overlap
	for m1 in range(max_run_num):
		for m2 in range(m1+1, max_run_num):
			for ev in range(eval_size):
				i = 0
				for nn_size in nn_sizes:
					overlap[full_run, ev, i] += len(np.intersect1d(nn_array[m1,ev,:nn_size],nn_array[m2,ev,:nn_size]))
					jaccard[full_run, ev, i] += len(np.intersect1d(nn_array[m1,ev,:nn_size],nn_array[m2,ev,:nn_size])) / len(np.union1d(nn_array[m1,ev,:nn_size],nn_array[m2,ev,:nn_size]))
					i += 1

	for m1 in range(max_run_num):
		for ev in range(eval_size):
			i = 0
			for nn_size in nn_sizes:
				tot = 0
				for j in range(nn_max_size):
					if j == nn_size - 1:
						continue
					distance_to_boundary = cos_array[m1,ev,j] - cos_array[m1,ev,nn_size - 1]
					probability = np.exp( - (distance_to_boundary * distance_to_boundary) / 1.11e-3)
					tot -= probability
				slope[full_run, ev,i] += tot
				i += 1

	overlap[full_run] /= (max_run_num + 1) * (max_run_num // 2)
	jaccard[full_run] /= (max_run_num + 1) * (max_run_num // 2)