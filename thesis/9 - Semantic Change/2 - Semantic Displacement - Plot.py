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

import pickle as pkl
import math
import numpy as np
import matplotlib as mpl 

mpl.rcParams.update({
	"axes.titlesize" : "medium" 
	})

import matplotlib.pyplot as plt

plt.rcParams.update({
	"pgf.texsystem": "pdflatex",
	"pgf.preamble": [
		 r"\usepackage[utf8x]{inputenc}",
		 r"\usepackage[T1]{fontenc}",
		 ]
})
pgf_with_rc_fonts = {
	"font.family": "serif",
	"font.serif": [],   
	"font.size": 10,                   
}

mpl.rcParams.update(pgf_with_rc_fonts)


random_exp_folder = '/home/rettenls/data/experiments/coha/random/'
genuine_exp_folder = '/home/rettenls/data/experiments/coha/'

ana_folder = '/home/rettenls/data/experiments/coha/analysis/'


# GET DATA

random_batches = ['batch_{:04d}'.format(i) for i in range(20)]
genuine_batches = [str(1810 + i * 10) for i in range(20)]


model = 'word2vec'
size = 32
batch_nums = [15,16,17,18]

for data_type in ['random', 'genuine']:
	
	if data_type == 'random':
		batches = random_batches
		exp_folder = random_exp_folder
	if data_type == 'genuine':
		batches = genuine_batches
		exp_folder = genuine_exp_folder

	frequencies = list()
	displacements = list()
	words = list()

	for batch_num in batch_nums:

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

		frequencies += freq
		displacements += cos_dist.tolist()

# PLOT

figsize = (5, 9)
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=figsize)

for i,cell in enumerate(ax):

	data_type = ['random', 'genuine'][i]
	color = ['blue', 'orange'][i]

	displacements = np.log(displacements)

	cell.scatter(frequencies, displacements, s = 0.2, c = color)
	cell.set_xscale('log')
	cell.set_xlim([1.e-5,1.e-1])

plt.show()
