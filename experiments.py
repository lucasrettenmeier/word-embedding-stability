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

text_folder = '/home/rettenls/data/texts/wiki/'
eval_folder = '/home/rettenls/data/evaluation/analogy/'
exp_folder = '/home/rettenls/data/experiments/wiki/'

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

languages = ['en'] #['fi', 'pt', 'zh', 'cs', 'pl', 'hi']#, 'it']#, 'es', 'fr', 'de', 'en']

models = ['word2vec', 'fasttext', 'glove']
model_types = {'word2vec': ['cbow', 'skipgram'], 'fasttext': ['skipgram'], 'glove': [None]}

data_types = ['fixed'] #['shuffle', 'bootstrap', 'fixed']

for language in languages:
	eval_file = eval_folder + 'questions-words-' + language + '.txt'
	for model in models:
		for model_type in model_types[model]:
			for data_type in data_types:

				# Determine folder name
				if model_type is None:
					folder = exp_folder + language + '/' + model + '/' + data_type 
				else:
					folder = exp_folder + language + '/' + model + '/' + model_type + '/' + data_type

				# Create folder if it doesn't exist
				if not os.path.isdir(folder):
					os.makedirs(folder)

				#---------------------------------------------------------------------------------------------------
				# NORMAL RUNS
				#---------------------------------------------------------------------------------------------------

				if (language == 'fi' and model == 'fasttext' and model_type == 'skipgram' and data_type == 'shuffle') or \
					(language == 'fi' and model == 'word2vec' and model_type == 'cbow' and data_type == 'shuffle'):
					max_run_num = 512
				else:
					max_run_num = 16

				for run_number in range(max_run_num):
		
					run = folder + '/run_{:04d}'.format(run_number)

					# Work to be done?
					if (not os.path.exists(run)) or (len(os.listdir(run)) < 5):
						# Already in progress
						print('IN:', run)
						if not run_in_progress(run + "_RUN"):

							try:

								# Model needs to be trained
								if (not os.path.exists(run)) or (len(os.listdir(run)) < 3):
									# Train & Save
									if data_type == 'fixed':
										text_file = text_folder + language +  '/fixed/run_0000.txt'
									else:
										text_file = text_folder + language +  '/' + data_type + \
											'/run_{:04d}.txt'.format(run_number)

									m = Model(model)
									m.train(text_file)
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
													text_file = text_folder + language +  '/' + data_type + \
														'/run_{:04d}.txt'.format(run_number)

													m = Model(model)
													m.train(text_file, epochs = ep, neg_samp_num = ns)
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

												# Evaluate
												models[k] = models[k].reduce(200000)
												models[k].normalize()
												evaluate_analogy(models[k], eval_file, eval_folder_name = run)

												
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
												text_file = text_folder + language +  '/' + data_type + \
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

				#---------------------------------------------------------------------------------------------------
				# CONVERGENCE EVALUATION RUNS
				#---------------------------------------------------------------------------------------------------
				if (os.path.isdir(folder + '/merge_nnz_{:04d}_run_0000'.format(max_run_num)) and \
					(len(os.listdir(folder + '/merge_nnz_{:04d}_run_0000'.format(max_run_num))) > 4) and \
					(not os.path.isdir(folder + '/convergence_analysis'))):
				
					# If run in progress -> Skip
					if not run_in_progress(folder + '_CONVERGENCE_NONNORMALIZED'):
					
						# Loop over averaging sizes: 2, 4, 8, etc.
						max_merge_num = int(math.log(max_run_num,2))
						merge_nums = np.arange(0,max_merge_num + 1)
						max_avg_num = 2 ** merge_nums[-1]

						if data_type != 'bootstrap':

							# Get Vocabulary Size
							m = Model(model)
							m.load(folder + '/run_0000')
							voc_size = m.voc_size
							del m

							# Randomly Reduce the Vocabulary
							reduced_count = int(5.e4)
							reduced_indices = np.arange(voc_size)
							np.random.shuffle(reduced_indices)
							reduced_indices = reduced_indices[:reduced_count]

						results = list()

						for merge_num in merge_nums:

							avg_size = 2 ** merge_num
							sample_size = min(max_avg_num // avg_size, 16)

							# Iterate over average samples

							merge_num_results = list()

							for m1_num in range(sample_size):
							
								if (avg_size == 1):
									m1_folder = folder + '/run_{:04d}'.format(m1_num)
								else:
									m1_folder = folder + '/merge_nnz_{:04d}_run_{:04d}'.format(avg_size,m1_num)

								m1 = Model(model)
								m1.load(m1_folder)
								m1.normalize()

								m1_results = list()

								for m2_num in range(m1_num,sample_size):

									if (avg_size == 1):
										m2_folder = folder + '/run_{:04d}'.format(m2_num)
									else:
										m2_folder = folder + '/merge_nnz_{:04d}_run_{:04d}'.format(avg_size,m2_num)

									m2 = Model(model)
									m2.load(m2_folder)
									m2.normalize()

									if data_type != 'bootstrap':
										m1, m2, _ = align(m1,m2)
										m1_results.append(get_pip_norm(m1,m2, word_indices = reduced_indices, reduced = True))

									else:
										m1, m2, joint = align(m1,m2)
										m1_results.append(get_pip_norm(m1,m2, word_indices = joint, reduced = True, get_proxy = True))

								print(m1_results)
								merge_num_results.append(m1_results)

							results.append(merge_num_results)

						if (not os.path.isdir(folder + '/convergence_analysis')):
							os.mkdir(folder + '/convergence_analysis')

						# Save voc to pickle file:
						with open(get_filename(folder + '/convergence_analysis', 'analysis', 'pkl'), 'wb') as handle:
							pickle.dump(results, handle, protocol = pickle.HIGHEST_PROTOCOL)