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

decades = [1900,1990]

models = ['word2vec']#, 'fasttext']#, 'glove']
model_types = {'word2vec': ['skipgram'], 'fasttext': [None], 'glove': [None]}

max_run_num = 4

for model in models:
	for model_type in model_types[model]:
			for decade in decades:
		
				# Determine folder name
				if model_type is None:
					folder = exp_folder + model + '/' + str(decade)
				else:
					folder = exp_folder + model + '/' + model_type + '/' + str(decade)

				# Create folder if it doesn't exist
				if not os.path.isdir(folder):
					os.makedirs(folder)

				#---------------------------------------------------------------------------------------------------
				# NORMAL RUNS
				#---------------------------------------------------------------------------------------------------

				for run_number in range(max_run_num):
		
					run = folder + '/run_{:04d}'.format(run_number)

					# Work to be done?
					if (not os.path.exists(run)) or (len(os.listdir(run)) < 3):
						# Already in progress
						if not run_in_progress(run + "_RUN"):

							# Train & Save
							m = Model(model)
							text_file = text_folder + 'shuffle/' + str(decade) + '/run_{:04d}.txt'.format(run_number)
							m.train(text_file)
							m.save(run)

"""
				
				#---------------------------------------------------------------------------------------------------
				# MERGE RUNS
				#---------------------------------------------------------------------------------------------------
				if (len(os.listdir(folder)) >= max_run_num):
				
					# If run in progress -> Skip
					if not run_in_progress(folder + '_MERGE_NONNORMALIZED'):
					
						# Loop over averaging sizes: 2, 4, 8, etc.
						max_merge_num = int(math.log(max_run_num,2))
						merge_nums = np.arange(1,max_merge_num + 1)
						max_avg_num = 2 ** merge_nums[-1]

						for merge_num in merge_nums:

							avg_size = 2 ** merge_num
							sample_size = max_avg_num // avg_size 

							# Iterate over average samples
							for sample_num in range(sample_size):
								
								try:

									run = folder + '/merge_nnz_{:04d}_run_{:04d}'.format(avg_size,sample_num)

									# Check if folder exists
									if (not os.path.exists(run)) or (len(os.listdir(run)) < 3): 
										m1 = Model(model)
										m2 = Model(model)

										base_avg_size = avg_size // 2
										base_sample_num = sample_num * 2

										if (base_avg_size == 1):
											base_run1 = folder + '/run_{:04d}'.format(base_sample_num)
											base_run2 = folder + '/run_{:04d}'.format(base_sample_num + 1)
										else:
											base_run1 = folder + '/merge_nnz_{:04d}_run_{:04d}'.format(base_avg_size,base_sample_num)
											base_run2 = folder + '/merge_nnz_{:04d}_run_{:04d}'.format(base_avg_size,base_sample_num + 1)

										m1.load(base_run1)
										m2.load(base_run2)

										m1, m2, joint = align(m1, m2)
										t = Transformation('orthogonal', train_at_init = True, model1 = m2, model2 = m1, joint = joint)
										print('Merge Level: {:2d},    Model Indices: ({:3d},{:3d}),     Cosine Similarity: {:.4f}'.format(merge_num, 
											base_sample_num, base_sample_num + 1, np.mean(get_cosine_similarity(m1, t.apply_to(m2), joint))))

										# Average
										m = avg(m1, t.apply_to(m2), joint_indices = joint, normalize = False)
										
										# Save
										m.save(run)

										# Evaluate
										m = m.reduce(200000)
										m.normalize()
										evaluate_analogy(m, eval_file, eval_folder_name = run)

									# Check if needs to be evaluated
									elif (len(os.listdir(run)) < 5):
										# Load
										m = Model(model)
										m.load(run)

										# Evaluate 
										m = m.reduce(200000)
										m.normalize()
										evaluate_analogy(m, eval_file, eval_folder_name = run)
								
								except:
									continue

"""