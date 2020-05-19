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
from scipy.stats		import spearmanr


#-------------------------------------------------------------------------------------------------------------------
# Experiments
#-------------------------------------------------------------------------------------------------------------------

language = 'fi'
model = 'word2vec'
model_type = 'skipgram'
data_type = 'shuffle'

total_run_num = 512

if model == 'glove':
	folder = exp_folder + language + '/' + model + '/' + data_type
else:
	folder = exp_folder + language + '/' + model + '/' + model_type + '/' + data_type

vals = list()

for run_num in range(total_run_num):
	m = Model(model)
	m.load(folder + '/run_{:04d}'.format(run_num))
	
	cs = get_word_relatedness(m, 'dog', 'cat')
	vals.append(cs)

	print(run_num, cs)