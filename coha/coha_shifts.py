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

#-------------------------------------------------------------------------------------------------------------------
# Checking the Coordination File
#-------------------------------------------------------------------------------------------------------------------

def run_in_progress(run):

	in_progress = False

	# Open file
	file = open(coordination_file, 'r+')
	lines = file.read().splitlines()
	file.seek(0)
	for line in lines:
		# Delete all lines which are older than 24 hours and all invalid lines
		try:
			line_datetime = datetime.datetime.strptime(line[-19:], date_format)
			if (line_datetime + datetime.timedelta(hours = 24) > datetime.datetime.now()):
				file.write(line + '\n')

				# If line is not older than 24 hours -> compare to run
				if (line[:len(run)] == run):
					in_progress = True
		except:
			continue

	if not (in_progress):
		file.write(run + '_DATETIME=' + datetime.datetime.now().strftime(date_format) + '\n')
		file.close()
		return False
	else:
		file.close()
		return True

#-------------------------------------------------------------------------------------------------------------------
# Experiments
#-------------------------------------------------------------------------------------------------------------------

decades = np.arange(1810, 2010, 10).tolist()

decades = [1900,1990]

models = ['fasttext', 'word2vec']#, 'glove']
model_types = {'word2vec': ['skipgram'], 'fasttext': [None], 'glove': [None]}


m0 = Model('fasttext')                                                                       
m0.load(exp_folder + '/fasttext/1900/run_0000')                                              


m1 = Model('fasttext')  
m1.load(exp_folder + '/fasttext/1990/run_0000')                                              

m0, m1, joint = align(m0,m1)

t = Transformation('orthogonal', train_at_init = True, model1 = m1, model2 = m0, joint = joint)

i = 0

indices = list()
for index in joint:
	if (m0.count[index] > m0.total_count * 1.e-5) and (m1.count[index] > m1.total_count * 1.e-5):
		indices.append(index)

indices = np.array(indices)
x = get_cosine_similarity(m0, t.apply_to(m1), indices)