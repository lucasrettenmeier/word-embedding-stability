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

text_folder = '/home/rettenls/data/texts/coha/'

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

decades = [int(1810 + i * 10) for i in range(20)]

models = ['fasttext']#, 'word2vec']#, 'glove']

max_run_num = 16

for model in models:
	frequencies = list()
	displacements = list()
	words = list()
	for decade in decades[:-1]:

		try:

			folder_1 = exp_folder + model + '/' + str(decade)
			folder_2 = exp_folder + model + '/' + str(decade + 10) 

			run_1 = folder_1 + '/run_{:04d}'.format(0)
			run_2 = folder_2 + '/run_{:04d}'.format(0)
				
			# Load models
			m1 = Model(model)
			m1.load(run_1)
			m2 = Model(model)
			m2.load(run_2)

			# Align
			m1,m2,joint = align(m1,m2)

			# Get all relevant words
			relevant_indices = [index for index in joint if (m1.count[index] > 500 and m2.count[index] > 500)]

			# Calculate their frequencies
			freq = [m1.count[index] / m1.total_count for index in relevant_indices]

			# Transform
			t = Transformation('orthogonal', train_at_init = True, model1 = m1, model2 = m2, joint = joint)
			m1 = t.apply_to(m1)

			# Calculate the displacement
			cos_dist = 1 - get_cosine_similarity(m1,m2, word_indices = relevant_indices)

			words += [m1.words[index] for index in relevant_indices]
			frequencies += freq
			displacements += cos_dist.tolist()

		except:
			continue

	w_arr = np.array(words)
	f_arr = np.array(frequencies)
	d_arr = np.array(displacements)

	# Log - Normalize
	f_arr = np.log(f_arr)
	f_arr -= np.mean(f_arr)
	f_arr /= np.std(f_arr)

	# Log - Normalize
	d_arr = np.log(d_arr)
	d_arr -= np.mean(d_arr)
	d_arr /= np.std(d_arr)

	break

	#plt.xscale('log')
	#plt.scatter(f_arr,d_arr)
	#plt.show()





	