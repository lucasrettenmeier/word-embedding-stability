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

text_folder = '/home/rettenls/data/experiments/semeval/texts/'
exp_folder = '/home/rettenls/data/experiments/semeval/experiments/'

coordination_file = exp_folder + 'coordination/coordinate.txt'

date_format = '%Y-%m-%d_%H:%M:%S'

#-------------------------------------------------------------------------------------------------------------------
# Loading own Modules
#-------------------------------------------------------------------------------------------------------------------

import sys
sys.path.append("/home/rettenls/code/")

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

languages = ['english', 'german', 'latin', 'swedish']
models = ['word2vec', 'fasttext', 'glove']
model_types = {'word2vec': ['skipgram'], 'fasttext': ['skipgram'], 'glove': [None]}
corpora = ['corpus1', 'corpus2']
data_types = ['bootstrap', 'shuffle']#, 'fixed']

for language in languages:
	for corpus in corpora:
		for model in models:
			for model_type in model_types[model]:
				for data_type in data_types:

					# Determine folder name
					if model_type is None:
						folder = exp_folder + language + '/' + corpus + '/' + model + '/' + data_type 
					else:
						folder = exp_folder + language + '/' + corpus + '/' + model + '/' + model_type + '/' + data_type

					# Create folder if it doesn't exist
					if not os.path.isdir(folder):
						os.makedirs(folder)

					#---------------------------------------------------------------------------------------------------
					# NORMAL RUNS
					#---------------------------------------------------------------------------------------------------
					
					max_run_num = 32

					for run_number in range(max_run_num):
			
						run = folder + '/run_{:04d}'.format(run_number)
						# Work to be done?
						if (not os.path.exists(run)) or (len(os.listdir(run)) < 3):
							# Already in progress
							print('IN:', run)
							if not run_in_progress(run + "_RUN"):

								#try:

								# Model needs to be trained
								if (not os.path.exists(run)) or (len(os.listdir(run)) < 3):
									# Train & Save
									if data_type == 'fixed':
										text_file = text_folder + language + '/' + corpus + '/fixed/original.txt'
									else:
										text_file = text_folder + language +  '/' + corpus + '/' + data_type + \
											'/run_{:04d}.txt'.format(run_number)

									m = Model(model)
									m.train(text_file)
									m.save(run)
	
								#except:
								#	continue

					
					#---------------------------------------------------------------------------------------------------
					# MERGE RUNS
					#---------------------------------------------------------------------------------------------------
					if (len(os.listdir(folder)) >= max_run_num):
					
						# If run in progress -> Skip
						if not run_in_progress(folder + '_MERGE'):
						
							# Loop over averaging sizes: 2, 4, 8, etc.
							max_merge_num = int(math.log(max_run_num,2))
							merge_nums = np.arange(1,max_merge_num + 1)
							max_avg_num = 2 ** merge_nums[-1]

							for merge_num in merge_nums:

								avg_size = 2 ** merge_num
								sample_size = max_avg_num // avg_size 

								# Iterate over average samples
								for sample_num in range(sample_size):
									
									run = folder + '/merge_{:04d}_run_{:04d}'.format(avg_size,sample_num)

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
											base_run1 = folder + '/merge_{:04d}_run_{:04d}'.format(base_avg_size,base_sample_num)
											base_run2 = folder + '/merge_{:04d}_run_{:04d}'.format(base_avg_size,base_sample_num + 1)
										
										try:
											m1.load(base_run1)
										except:
											shutil.rmtree(base_run1)
											break

										try:
											m2.load(base_run2)
										except:
											shutil.rmtree(base_run2)
											break
										

										m1, m2, joint = align(m1, m2)
										t = Transformation('orthogonal', train_at_init = True, model1 = m2, model2 = m1, joint = joint)
										print('Merge Level: {:2d},    Model Indices: ({:3d},{:3d}),     Cosine Similarity: {:.4f}'.format(merge_num, 
											base_sample_num, base_sample_num + 1, np.mean(get_cosine_similarity(m1, t.apply_to(m2), joint))))

										# Average
										m = avg(m1, t.apply_to(m2), joint_indices = joint, normalize = False)
										
										# Save
										m.save(run)


					"""
					#---------------------------------------------------------------------------------------------------
					# LONG RUNS
					#---------------------------------------------------------------------------------------------------
					if (data_type == 'shuffle'):
						if (model != 'glove'):
							for run_number in range(4):
								for ep in [5,10,20,40]:
									for ns in [5,10,20]:
										if ep == 5 and ns == 5:
											continue

										run = folder + '/ep_{:04d}_ns_{:04d}_run_{:04d}'.format(ep, ns, run_number)

										# Work to be done?
										if (not os.path.exists(run)) or (len(os.listdir(run)) < 5):
											# Already in progress?
											if not run_in_progress(run):

												try:

													# Model needs to be trained
													if (not os.path.exists(run)) or (len(os.listdir(run)) < 3):
														text_file = text_folder + language +  '/' + corpus +  '/' + data_type + \
															'/run_{:04d}.txt'.format(run_number)

														m = Model(model)
														m.train(text_file, epochs = ep, neg_samp_num = ns)
														m.save(run)

													# Model already trained
													else:
														# Load
														m = Model(model)
														m.load(run)

												except:
													continue

										if (run_number == 3): # MERGE

											print('\n\nMERGE\n\n')

											run = folder + '/ep_{:04d}_ns_{:04d}_merge'.format(ep, ns)

											# Work to be done?
											if (not os.path.exists(run)) or (len(os.listdir(run)) < 5):
												# Already in progress?
												if not run_in_progress(run):
											
													# Average Models
													models = list()
													for i in range(4):
														m = Model(model)
														run = folder + '/ep_{:04d}_ns_{:04d}_run_{:04d}'.format(ep, ns, i)
														m.load(run)
														models.append(m)

													for i in range(2):
														k = i * 2
														l = i * 2 + 1
														models[k], models[l], joint = align(models[k], models[l])
														t = Transformation('orthogonal', train_at_init = True, model1 = models[l], model2 = models[k], joint = joint)
														models[k] = avg(models[k], t.apply_to(models[l]), joint_indices = joint, normalize = False)

													k = 0
													l = 2
													models[k], models[l], joint = align(models[k], models[l])
													t = Transformation('orthogonal', train_at_init = True, model1 = models[l], model2 = models[k], joint = joint)
													models[k] = avg(models[k], t.apply_to(models[l]), joint_indices = joint, normalize = False)

													# Save
													run = folder + '/ep_{:04d}_ns_{:04d}_merge'.format(ep, ns)
													models[k].save(run)
													
						else:
							for run_number in range(4):
								for ep in [200,400]:
									run = folder + '/ep_{:04d}_run_{:04d}'.format(ep, run_number)

									# Work to be done?
									if (not os.path.exists(run)) or (len(os.listdir(run)) < 5):
										# Already in progress?
										if not run_in_progress(run):

											try:

												# Model needs to be trained
												if (not os.path.exists(run)) or (len(os.listdir(run)) < 3):
													text_file = text_folder + language +  '/' + corpus +  '/' + data_type + \
														'/run_{:04d}.txt'.format(run_number)

													m = Model(model)
													m.train(text_file, epochs = ep)
													m.save(run)

												# Model already trained
												else:
													# Load
													m = Model(model)
													m.load(run)

												# Evaluate
												m = m.reduce(200000)
												m.normalize()
												evaluate_analogy(m, eval_file, eval_folder_name = run)

											except:
												continue

									if (run_number == 3): # MERGE

										print('\n\nMERGE\n\n')

										run = folder + '/ep_{:04d}_merge'.format(ep)

										# Work to be done?
										if (not os.path.exists(run)) or (len(os.listdir(run)) < 5):
											# Already in progress?
											if not run_in_progress(run):
										
												# Average Models
												models = list()
												for i in range(4):
													m = Model(model)
													run = folder + '/ep_{:04d}_run_{:04d}'.format(ep, i)
													m.load(run)
													models.append(m)

												for i in range(2):
													k = i * 2
													l = i * 2 + 1
													models[k], models[l], joint = align(models[k], models[l])
													t = Transformation('orthogonal', train_at_init = True, model1 = models[l], model2 = models[k], joint = joint)
													models[k] = avg(models[k], t.apply_to(models[l]), joint_indices = joint, normalize = False)

												k = 0
												l = 2
												models[k], models[l], joint = align(models[k], models[l])
												t = Transformation('orthogonal', train_at_init = True, model1 = models[l], model2 = models[k], joint = joint)
												models[k] = avg(models[k], t.apply_to(models[l]), joint_indices = joint, normalize = False)

												# Save
												run = folder + '/ep_{:04d}_merge'.format(ep)
												models[k].save(run)

												# Evaluate
												models[k] = models[k].reduce(200000)
												models[k].normalize()
												evaluate_analogy(models[k], eval_file, eval_folder_name = run)
					"""