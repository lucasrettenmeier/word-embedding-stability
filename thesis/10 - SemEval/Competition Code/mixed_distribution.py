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
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

# Writing Output
import pickle

exp_folder = '/home/rettenls/data/experiments/semeval/'

#-------------------------------------------------------------------------------------------------------------------
# Loading own Modules
#-------------------------------------------------------------------------------------------------------------------

import sys
sys.path.append("/home/rettenls/code/")

from lib.model 			import Model
from lib.trafo 			import Transformation
from lib.eval 			import print_nn_word, get_nn_list, get_cosine_similarity, get_pip_norm, get_ww_pip_norm
from lib.score 			import evaluate_analogy
from lib.operations 	import align, avg, align_list
from lib.util			import get_filename

#-------------------------------------------------------------------------------------------------------------------
# Checking the Coordination File
#-------------------------------------------------------------------------------------------------------------------

languages = ['english', 'latin', 'german', 'swedish']
corpora = ['corpus1', 'corpus2']

models = ['fasttext', 'word2vec', 'glove']
model_types = {'fasttext': ['skipgram'], 'word2vec' : ['skipgram'], 'glove' : [None]}

max_run_num = 16

results = dict()

for language in languages:
	for model in models:

		m1 = Model(model)
		if model != 'glove':
			m1_folder = exp_folder + 'experiments/' + language + '/corpus1/' + model + '/skipgram/shuffle/merge_0016_run_{:04d}'.format(0)
		else:
			m1_folder = exp_folder + 'experiments/' + language + '/corpus1/' + model + '/shuffle/merge_0016_run_{:04d}'.format(0)
		m1.load(m1_folder)

		m2 = Model(model)
		if model != 'glove':
			m2_folder = exp_folder + 'experiments/' + language + '/corpus2/' + model + '/skipgram/shuffle/merge_0016_run_{:04d}'.format(0)
		else:
			m2_folder = exp_folder + 'experiments/' + language + '/corpus2/' + model + '/shuffle/merge_0016_run_{:04d}'.format(0)
		m2.load(m2_folder)

		eval_word_list = pickle.load(open(exp_folder + 'distributions/' + language + '/words.pickle', 'rb'))
		m1,m2,joint = align(m1,m2)
		eval_indices = [m1.indices[w] for w in eval_word_list]
		results[model] = get_ww_pip_norm(model1 = m1, model2 = m2, eval_indices = eval_indices, avg_indices = joint)

	# Save Results and Word List
	dist_folder = exp_folder + 'mixed_distributions/' + language +'/'	
	pickle.dump(results, open(dist_folder + 'results.pickle', 'wb'))
	pickle.dump(eval_word_list, open(dist_folder + 'words.pickle', 'wb'))
