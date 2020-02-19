#-------------------------------------------------------------------------------------------------------------------
# Packages & Settings
#-------------------------------------------------------------------------------------------------------------------

# General packages
import time
import sys
import os
import datetime
from glob import glob
import shutil
import copy

# Math and data structure packages
import numpy as np
from scipy import stats
import math

# Writing Output
import pickle

text_folder = '/home/rettenls/data/texts/wiki/'
eval_folder = '/home/rettenls/data/evaluation/analogy/'
exp_folder = '/home/rettenls/data/experiments/wiki/'

coordination_file = exp_folder + 'coordination/coordinate.txt'

date_format = '%Y-%m-%d_%H:%M:%S'

#-------------------------------------------------------------------------------------------------------------------
# Loading own Modules
#-------------------------------------------------------------------------------------------------------------------

from lib.model 			import Model
from lib.trafo 			import Transformation
from lib.eval 			import print_nn_word, get_nn_list, get_cosine_similarity, get_pip_norm
from lib.score 			import evaluate_analogy
from lib.operations 	import align, avg
from lib.util			import get_filename


models = ['fasttext']#, 'word2vec']
model_types = {'word2vec': ['skipgram'], 'fasttext': ['skipgram']}
data_types = ['shuffle']#, 'bootstrap', 'fixed']

language = 'fi'

max_run_num = 10
	
for model in models:
	for model_type in model_types[model]:
		for data_type in data_types:

			folder = exp_folder + language + '/' + model + '/' + model_type + '/' + data_type

			avg = None

			for run_number in range(max_run_num):

				run = folder + '/run_{:04d}'.format(run_number)
				m = Model(model)
				m.load(run)

				# Add to Average
				if (avg is None):
					avg = copy.deepcopy(m)
				else:
					avg, m, joint = align(avg, m)
					avg.embeddings += m.embeddings
