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
source_texts = '/home/rettenls/data/texts/coha/source/Texts/'
batch_texts = '/home/rettenls/data/texts/coha/random/texts/'


exp_folder = '/home/rettenls/data/experiments/coha/'


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

max_run_num = 16

# Get all Files
file_paths = list()
for decade in decades:
	path = source_texts + str(decade) + '/'
	file_paths += [path + x for x in os.listdir(path)]

# Shuffle
path_arr = np.array(file_paths)
np.random.shuffle(path_arr)

# Split
batch_num = 20
batch_size = (len(path_arr) // batch_num) + 1

upper = 0
for bn in range(batch_num):
	try:
		os.mkdir(batch_texts + 'batch_{:04d}'.format(bn))
	except:
		pass

	lower = upper
	upper = min(len(path_arr), lower + batch_size)

	paths = path_arr[lower:upper]

	fn = 0
	for path in paths:
		shutil.copyfile(path, batch_texts + 'batch_{:04d}/file_{:04d}'.format(bn, fn))
		fn += 1