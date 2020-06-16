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

# Math and data structure packages
import numpy as np
from scipy import stats
import math

# Writing Output
import pickle

text_folder = '/home/rettenls/data/texts/coha/'
exp_folder = '/home/rettenls/data/experiments/coha/'
rand_folder = '/home/rettenls/data/experiments/coha/random/' 


coordination_file = exp_folder + 'coordination/coordinate.txt'

date_format = '%Y-%m-%d_%H:%M:%S'

#-------------------------------------------------------------------------------------------------------------------
# Loading own Modules
#-------------------------------------------------------------------------------------------------------------------

sys.path.append('/home/rettenls/code')
from lib.model 			import Model
from lib.trafo 			import Transformation
from lib.eval 			import print_nn_word, get_nn_list, get_cosine_similarity, get_pip_norm
from lib.score 			import evaluate_analogy
from lib.operations 	import align, avg
from lib.util			import get_filename


#-------------------------------------------------------------------------------------------------------------------
# Experiments
#-------------------------------------------------------------------------------------------------------------------

batches = ['batch_{:04d}'.format(i) for i in range(20)]
decades = [str(1810 + 10 * i) for i in range(20)]
models = ['fasttext', 'word2vec', 'glove']
max_run_num = 32

for model in models:
	for decade in decades:
		folder = exp_folder + model + '/' + decade
		for run_folder in os.listdir(folder):
			run = folder + '/' + run_folder
			m = Model(model)
			try:
				m.load(run)
			except:
				shutil.rmtree(run)

for model in models:
	for batch in batches:
		rand_folder = exp_folder + model + '/' + batch
		for run_folder in os.listdir(folder):
			run = folder + '/' + run_folder
			m = Model(model)
			try:
				m.load(run)
			except:
				shutil.rmtree(run)



