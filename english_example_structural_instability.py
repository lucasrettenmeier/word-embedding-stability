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

language = 'en'
model = 'fasttext'
model_type = 'skipgram'
data_type = 'shuffle'

nn_sizes = [5,10]
nn_max_size = 50
eval_size = 1000

folder = exp_folder + language + '/' + model + '/' + model_type + '/correct_lr_original_preproc'

m = Model(model)
m.load(folder)

return_val = get_nn_list(m, m, np.arange(eval_size), nn_max_size, False, True)
nn_array = np.array(return_val[0])
cos_array = np.array(return_val[1])

structural_inst = np.zeros((eval_size,len(nn_sizes)))
i = 0
for nn_size in nn_sizes:
	for ev in range(eval_size):
		tot = 0
		for j in range(nn_max_size):
			if j == nn_size - 1:
				continue
			distance_to_boundary = cos_array[ev,j] - cos_array[ev,nn_size - 1]
			probability = np.exp( - (distance_to_boundary * distance_to_boundary) / 1.11e-3)
			tot -= probability
		structural_inst[ev,i] += tot
	i += 1
