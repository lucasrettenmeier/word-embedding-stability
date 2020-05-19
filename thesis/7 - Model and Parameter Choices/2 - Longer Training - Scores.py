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

date_format = '%Y-%m-%d_%H:%M:%S'

#-------------------------------------------------------------------------------------------------------------------
# Loading own Modules
#-------------------------------------------------------------------------------------------------------------------

sys.path.append('/home/rettenls/code')
from lib.model 			import Model
from lib.trafo 			import Transformation
from lib.eval 			import print_nn_word, get_nn_list, get_cosine_similarity, get_pip_norm, get_word_relatedness,get_common_vocab
from lib.score 			import evaluate_analogy
from lib.operations 	import align, avg, join, align_list
from lib.util			import get_filename
from scipy.stats		import spearmanr


#-------------------------------------------------------------------------------------------------------------------
# Experiments
#-------------------------------------------------------------------------------------------------------------------


text_folder = '/home/rettenls/data/texts/wiki/'
eval_folder = '/home/rettenls/data/evaluation/analogy/'
exp_folder = '/home/rettenls/data/experiments/wiki/'

languages = ['hi', 'fi', 'zh', 'cs', 'pl', 'pt']

models = ['fasttext', 'glove', 'word2vec']
models = ['fasttext', 'word2vec']
model_type = 'skipgram'

data_types = ['shuffle']

epochs = [5,10,20,40]
ns_sizes = [5,10,20]

total_run_number = 4

for language in languages:
	for model in models:
		for data_type in data_types:

			# Get Folder
			if model == 'glove':
				folder = exp_folder + language + '/' + model + '/' + data_type
			else:
				folder = exp_folder + language + '/' + model + '/' + model_type + '/' + data_type
		
			for epoch in epochs:
				for ns in ns_sizes:
					results = list()
					for run_number in range(total_run_number):
						try:
							if ns == 5 and epoch == 5:
								f = open(folder + '/run_{:04d}/eval.txt'.format(run_number))
							else:
								f = open(folder + '/ep_{:04d}_ns_{:04d}_run_{:04d}/eval.txt'.format(epoch, ns, run_number))
							lines = f.readlines()
							results.append(float(lines[7][-8:-3].replace('(','')))
						except:
							pass

					print(language, model, epoch, ns, len(results), np.mean(results), np.std(results))

