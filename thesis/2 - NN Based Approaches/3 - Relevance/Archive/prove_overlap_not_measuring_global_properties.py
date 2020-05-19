#-------------------------------------------------------------------------------------------------------------------
# Packages & Settings
#-------------------------------------------------------------------------------------------------------------------

# General packages
import time
import sys
import os
import datetime
from glob import glob
import copy

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
from lib.operations 	import align, avg, join, align_list, get_common_vocab, k_means_clustering
from lib.util			import get_filename


#-------------------------------------------------------------------------------------------------------------------
# Experiments
#-------------------------------------------------------------------------------------------------------------------

language = 'cs'
model = 'word2vec'
model_type = 'skipgram'
data_type = 'shuffle'

if model == 'glove':
		folder = exp_folder + language + '/' + model + '/' + data_type
else:
	folder = exp_folder + language + '/' + model + '/' + model_type + '/' + data_type

m1 = Model(model)
m1.load(folder + '/run_{:04d}'.format(0))
#m2 = copy.deepcopy(m1)

m2 = Model(model)
m2.load(folder + '/run_{:04d}'.format(4))

m1, m2, _ = align(m1,m2)
"""
k = 10
cluster_array, center_array = k_means_clustering(m1, k)


avg_vals = list()
for i in range(k):
	line = ''
	avg = 0
	for j in range(k):
		val = np.matmul(center_array[i],center_array[j].T)
		avg += val
		line += '{:.2f} '.format(val)
	avg_vals.append(avg / 10)
	print( '{:.2f} '.format(avg / 10), ' | ', line)

for i in np.argsort(avg_vals)[:5]:
	print(i, avg_vals[i])
	m2.embeddings[np.where(cluster_array == i)] = m2.embeddings[np.where(cluster_array == i)] * (-1)

"""

eval_size = 1000
eval_indices = np.arange(m1.voc_size)
np.random.shuffle(eval_indices)
eval_indices = eval_indices[:eval_size]

nn_list_1 = get_nn_list(m1, m1, eval_indices, 10, False, True)[0]
nn_list_2 = get_nn_list(m2, m2, eval_indices, 10, False, True)[0]

total_ov = 0
for ev in range(eval_size):
	total_ov += (len(np.intersect1d(nn_list_1[ev],nn_list_2[ev])))

print('\nOverlap: {:.1f}%'.format(total_ov * 10 / eval_size))

eval_folder = '/home/rettenls/data/evaluation/analogy/'
eval_file = eval_folder + 'questions-words-' + language + '.txt'

m1 = m1.reduce(200000)
m1.normalize()
semantic_questions_1, syntactic_questions_1, semantic_responses_1, syntactic_responses_1 = evaluate_analogy(m1, eval_file)

m2 = m2.reduce(200000)
m2.normalize()
semantic_questions_2, syntactic_questions_2, semantic_responses_2, syntactic_responses_2 = evaluate_analogy(m2, eval_file)