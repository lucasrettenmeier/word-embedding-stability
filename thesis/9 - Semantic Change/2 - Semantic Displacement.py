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

random_exp_folder = '/home/rettenls/data/experiments/coha/random/'
genuine_exp_folder = '/home/rettenls/data/experiments/coha/'

ana_folder = '/home/rettenls/data/experiments/coha/analysis/'

random_batches = ['batch_{:04d}'.format(i) for i in range(20)]
genuine_batches = [str(1810 + i * 10) for i in range(20)]
models = ['word2vec', 'fasttext', 'glove']
max_size = 32
sizes = [2 ** level for level in range(int(math.log(max_size,2)) + 1)]

for data_type in ['random', 'genuine']:
	for model in models:
		if data_type == 'random':
			batches = random_batches
			exp_folder = random_exp_folder
		if data_type == 'genuine':
			batches = genuine_batches
			exp_folder = genuine_exp_folder

		for size in sizes:
	
			if(os.path.exists(ana_folder + model + '_' + data_type + '_{:04d}'.format(size))):
				continue		

			frequencies = list()
			displacements = list()
			words = list()

			for batch_num in range(len(batches) - 1):

				folder_1 = exp_folder + model + '/' + batches[batch_num]
				folder_2 = exp_folder + model + '/' + batches[batch_num + 1]

				if size == 1:
					run_1 = folder_1 + '/run_{:04d}'.format(0)
					run_2 = folder_2 + '/run_{:04d}'.format(0)
				else:
					run_1 = folder_1 + '/merge_nnz_{:04d}_run_{:04d}'.format(size, 0)
					run_2 = folder_2 + '/merge_nnz_{:04d}_run_{:04d}'.format(size, 0)
					
				# Load models
				m1 = Model(model)
				m1.load(run_1)
				m2 = Model(model)
				m2.load(run_2)

				# Align
				m1,m2,joint = align(m1,m2)

				# Get all relevant words
				relevant_indices = [index for index in joint if (m1.count[index] >= 501 * size and m2.count[index] >= 501 * size)]

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

			r_arr = np.stack((w_arr, f_arr, d_arr), axis = 1)
			np.savez(open(ana_folder + model + '_' + data_type + '_{:04d}'.format(size), 'wb'), words = w_arr, displacements = d_arr, frequencies = f_arr)