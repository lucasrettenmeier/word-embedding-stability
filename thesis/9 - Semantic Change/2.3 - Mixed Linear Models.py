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

# Writing Output
import pickle

# Regression
from statsmodels.regression import mixed_linear_model 

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


ana_folder = '/home/rettenls/data/experiments/coha/analysis/'
models = ['word2vec']#, 'fasttext']#, 'glove']
max_size = 32
sizes = [32]

result = dict()

for model in models:
	for data_type in ['genuine', 'random']:
		res = list()
		for size in sizes:
			data = np.load(open(ana_folder + model + '_' + data_type + '_{:04d}'.format(size), 'rb'))

			words = data['words']
			frequencies = data['frequencies']
			displacements = data['displacements']

			md = mixed_linear_model.MixedLM(endog = displacements, exog = frequencies, groups = words)
			rs = md.fit()
			print(rs.summary())
			res.append(rs.params[0])
			print(rs.params)


		#print(res)
		#result[model + '_' + data_type] = np.array(res)