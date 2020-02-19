#-------------------------------------------------------------------------------------------------------------------
# Packages & Settings
#-------------------------------------------------------------------------------------------------------------------

# General
import os
import datetime
import time
import copy as cp

# Languages / Encodings
import codecs

# Math and Data Structure
import numpy as np
import scipy.stats as stat
#-------------------------------------------------------------------------------------------------------------------
# Loading own Modules
#-------------------------------------------------------------------------------------------------------------------

from lib.model 	import Model
from lib.eval 	import get_nn_list, get_word_relatedness
from lib.util 	import print_log_line, write_log_line, get_filename

#-------------------------------------------------------------------------------------------------------------------
# GET ANALOGY EVALUATION SCORES
#
# Takes a model and an evaluation file and 
#-------------------------------------------------------------------------------------------------------------------

def evaluate_analogy(model, read_eval_file, eval_folder_name = None, prediction_mode = 'linear', 
	lowercase_questions = True):
	
	print('\nEvaluation of word embedding model started ...')

	#---------------------------------------------------------------------------------------------------------------
	# Prerequsite: Transform model to lowercase if all questions are lowercase
	#
	# Build index dictionary by iterating over all words from the least frequent to the most frequent ones
	# this is necessary to ensure model.indices[word] yields the index of the most commonly used word for a 
	# given lowercased word.
	#
	# Example: Word1: Vietnam, Index1: 1000, Count1: 2.e5; Word2: vietnam, Index2: 180000, Count2: 1.e2
	# After lowercasing, model.indices['vietnam'] will yield 1000, not 180000. 
	#---------------------------------------------------------------------------------------------------------------

	if (lowercase_questions):
		model = cp.deepcopy(model)
		for index, word in model.words.items():
			model.words[index] = word.lower()

		model.indices = dict()
		for index in np.argsort(model.count):
			model.indices[model.words[index]] = index 

	#---------------------------------------------------------------------------------------------------------------
	# PART I - PREPROCESSING: Read the file, get the total number of questions, differentiate between syntactic and 
	# 	semantic questions and remove invalid questions (containing words not occuring in the vocabulary).
	#---------------------------------------------------------------------------------------------------------------

	# Read file
	read_eval_file = os.path.abspath(read_eval_file)
	file = codecs.open(read_eval_file, 'r', encoding = 'utf-8')
	lines = file.readlines()
	file.close()

	# Lists for syntactic and semantic questions:	
	semantic_questions = []
	syntactic_questions = []
	mode = 'None'

	# Number of invalid questions
	invalid_question_num = 0

	# Loop over lines
	for line in lines:
		# Evaluate meta-data lines:
		if(line[0] == ':'):
			if (line[:6] == ': gram'):
				mode = 'syntactic'
			else:
				mode = 'semantic'

		else:
			
			if (lowercase_questions):
				words = line.lower().strip('\n').split(' ')
			else:
				words = line.strip('\n').split(' ')

			# Check if the question is valid (all words in model.indices):
			valid = True
			for word in words:
				if word not in model.indices:
					valid = False
					break

			if valid:
				if (mode == 'syntactic'):
					syntactic_questions.append(words)
				if (mode == 'semantic'):
					semantic_questions.append(words)
			else:
				invalid_question_num += 1

	semantic_num = len(semantic_questions)
	syntactic_num = len(syntactic_questions)

	words_per_question = len(semantic_questions[0])

	print('Completed step 1: Initialization. Found {} semantic, {} syntactic and {} invalid questions.'.format(
		semantic_num, syntactic_num, invalid_question_num))

	#---------------------------------------------------------------------------------------------------------------
	# PART II - PREDICTION: Loop over valid questions.
	#---------------------------------------------------------------------------------------------------------------

	# 3D array to store all vectors
	semantic_vectors = np.empty((semantic_num, words_per_question, model.dim_num))
	syntactic_vectors = np.empty((syntactic_num, words_per_question, model.dim_num))

	# Loop over questions
	line_num = 0
	for question in semantic_questions:
		word_num = 0
		for word in question:
			semantic_vectors[line_num,word_num,:] = model.embeddings[model.indices[word]]
			word_num += 1
		line_num += 1

	# Loop over questions
	line_num = 0
	for question in syntactic_questions:
		word_num = 0
		for word in question:
			syntactic_vectors[line_num,word_num,:] = model.embeddings[model.indices[word]]
			word_num += 1
		line_num += 1

	# Prediction model
	semantic_prediction = Model('generic')
	syntactic_prediction = Model('generic')

	# LINEAR PREDICTION
	if(prediction_mode == 'linear'):
		semantic_prediction.embeddings = (semantic_vectors[:,1,:] - semantic_vectors[:,0,:] 
			+ semantic_vectors[:,2,:])
		syntactic_prediction.embeddings = (syntactic_vectors[:,1,:] - syntactic_vectors[:,0,:] 
			+ syntactic_vectors[:,2,:])

	# Timing
	step2_time = time.time()

	print('Completed step 2: Prediction.')
	print('Started NN Analysis. This might take a while...')

	#---------------------------------------------------------------------------------------------------------------
	# PART III - EVALUATION: NN Analysis of the predicted vectors
	#---------------------------------------------------------------------------------------------------------------
	if(semantic_num > 0):
		semantic_nns = get_nn_list(model1 = model, model2 = semantic_prediction, word_indices = np.arange(semantic_num), 
			k = words_per_question, allow_self_neighbor = True, get_cosine_values = False)

	if(syntactic_num > 0):
		syntactic_nns = get_nn_list(model1 = model, model2 = syntactic_prediction, word_indices = np.arange(syntactic_num),
			k = words_per_question, allow_self_neighbor = True, get_cosine_values = False)

	# Timing
	step3_time = time.time()
	step3_duration = step3_time - step2_time

	print('Completed step 3: NN Analysis of predicted vectors. Total time:	{0} seconds.'.format(int(step3_duration)))

	# Remove the indices of the three input words 
	line_num = 0
	for question in semantic_questions:
		for word in question[:3]:
			wi = model.indices[word]
			if wi in semantic_nns[line_num]: semantic_nns[line_num].remove(wi)
		line_num += 1
	
	line_num = 0
	for question in syntactic_questions:
		for word in question[:3]:
			wi = model.indices[word]
			if wi in syntactic_nns[line_num]: syntactic_nns[line_num].remove(wi)
		line_num += 1

	# Evaluate the model:
	semantic_correct = 0
	syntactic_correct = 0

	semantic_responses = list()
	syntactic_responses = list()

	if(semantic_num > 0):
		line_num = 0
		for response in semantic_nns:
			word = model.words[response[0]]
			semantic_responses.append(word)
			if word == semantic_questions[line_num][-1]:
				semantic_correct += 1
			line_num += 1
	else: 
		semantic_correct = 0

	if(syntactic_num > 0):
		line_num = 0
		for response in syntactic_nns:
			word = model.words[response[0]]
			syntactic_responses.append(word)
			if word == syntactic_questions[line_num][-1]:
				syntactic_correct += 1
			line_num += 1
	else:
		syntactic_correct = 0

	#---------------------------------------------------------------------------------------------------------------
	# OUTPUT
	#---------------------------------------------------------------------------------------------------------------

	# Calculate all Scores:
	iv_num = invalid_question_num
	qu_num = invalid_question_num + semantic_num + syntactic_num
	iv_pct = invalid_question_num / (invalid_question_num + semantic_num + syntactic_num)

	se_num = semantic_num
	se_cor = semantic_correct
	if (semantic_num > 0):
		se_scr = semantic_correct / semantic_num
	else:
		se_scr = 0

	sy_num = syntactic_num
	sy_cor = syntactic_correct
	if (syntactic_num > 0):
		sy_scr = syntactic_correct / syntactic_num
	else:
		sy_scr = 0

	to_num = semantic_num + syntactic_num
	to_cor = semantic_correct + syntactic_correct
	to_scr = (semantic_correct + syntactic_correct) / (semantic_num + syntactic_num)


	if eval_folder_name is not None:

		# Handle filepath
		eval_folder_name = os.path.abspath(eval_folder_name)
		
		# Save evaluation to numpy file
		np.savez_compressed(get_filename(eval_folder_name, 'eval', 'npz'), semantic_questions = semantic_questions,
			syntactic_questions = syntactic_questions, semantic_responses = semantic_responses,
			syntactic_responses = syntactic_responses)

		# Logfile: General
		log = open(get_filename(eval_folder_name, 'eval', 'txt'), 'w')
		log.write('Evaluation of Word Embeddings.\n'
				'Implementation by Lucas Rettenmeier, Heidelberg Institute for Theoretical Studies.\n\n')

		write_log_line(log, 'Created on', datetime.datetime.now() , 36)

		# Logfile: Evaluation Results
		log.write('{:36s} {:5d} / {:5d}    ({:.2%})\n'.format(str('Invalid questions:'), iv_num, qu_num, iv_pct))
		log.write('{:36s} {:5d} / {:5d}    ({:.2%})\n'.format(str('Score on semantic questions:'), se_cor, se_num, se_scr))
		log.write('{:36s} {:5d} / {:5d}    ({:.2%})\n'.format(str('Score on syntactic questions:'), sy_cor, sy_num, sy_scr))
		log.write('{:36s} {:5d} / {:5d}    ({:.2%})\n'.format(str('Total score:'), to_cor, to_num, to_scr))

		# Logfile: Model Details
		log.write('\nEmbedding Model Details:\n')
		write_log_line(log, 'Embedding Model', model.model_type, 36)
		write_log_line(log, 'Number of Dimensions', model.dim_num , 36)
		write_log_line(log, 'Vocabulary Size', model.voc_size , 36)
		write_log_line(log, 'Total Word Count', model.total_count , 36)
		for key in model.training_parameters:
			write_log_line(log, key, model.training_parameters[key], 36)
		log.close()

	#---------------------------------------------------------------------------------------------------------------
	# CONSOLE LOG
	#---------------------------------------------------------------------------------------------------------------
	print('\nCompleted evaluation of word embeddings.')
	print('{:36s} {:5d} / {:5d}    ({:.2%})'.format(str('Invalid questions:'), iv_num, qu_num, iv_pct))
	print('{:36s} {:5d} / {:5d}    ({:.2%})'.format(str('Score on semantic questions:'), se_cor, se_num, se_scr))
	print('{:36s} {:5d} / {:5d}    ({:.2%})'.format(str('Score on syntactic questions:'), sy_cor, sy_num, sy_scr))
	print('{:36s} {:5d} / {:5d}    ({:.2%})'.format(str('Total score:'), to_cor, to_num, to_scr))
	if eval_folder_name is not None: print_log_line('Saved evaluation to folder:', eval_folder_name, 36)
	print('')

	return semantic_questions, syntactic_questions, semantic_responses, syntactic_responses

#-------------------------------------------------------------------------------------------------------------------
# GET SIMILARITY EVALUATION SCORES
#
# Takes a model and an evaluation file and 
#-------------------------------------------------------------------------------------------------------------------

def evaluate_similarity(model, read_eval_file, warnings = False, eval_folder_name = None, 
	lowercase_questions = True):

	#---------------------------------------------------------------------------------------------------------------
	# Prerequsite: Transform model to lowercase if all questions are lowercase
	#
	# Build index dictionary by iterating over all words from the least frequent to the most frequent ones
	# this is necessary to ensure model.indices[word] yields the index of the most commonly used word for a 
	# given lowercased word.
	#
	# Example: Word1: Vietnam, Index1: 1000, Count1: 2.e5; Word2: vietnam, Index2: 180000, Count2: 1.e2
	# After lowercasing, model.indices['vietnam'] will yield 1000, not 180000. 
	#---------------------------------------------------------------------------------------------------------------

	if (lowercase_questions):
		model = cp.deepcopy(model)
		for index, word in model.words.items():
			model.words[index] = word.lower()

		model.indices = dict()
		for index in np.argsort(model.count):
			model.indices[model.words[index]] = index 
	

	# Read file
	read_eval_file = os.path.abspath(read_eval_file)
	file = open(read_eval_file, 'r', encoding = 'utf-8')
	lines = file.readlines()
	file.close()

	tasks = list()

	human_judgement = list()
	vector_correlation = list()

	invalid_count = 0 

	for line in lines[1:]:
		# Get words and relatedness
		if (lowercase_questions):
			line = line.lower().split(',')[:3]
		else:
			line = line.split(',')[:3]
		
		# Append to task list
		tasks.append([line[0], line[1]])

		if (line[0] in model.indices) and (line[1] in model.indices):

			# Get human judgement and vector correlation
			human_judgement.append(float(line[2]))

			cosine = get_word_relatedness(model, tasks[-1][0], tasks[-1][1], warnings)
			vector_correlation.append(cosine)
			
			if cosine == 0:
				invalid_count += 1


	result = stat.spearmanr(human_judgement, vector_correlation)
		
	if eval_folder_name is not None:

		# Handle filepath
		eval_folder_name = os.path.abspath(eval_folder_name)

		# Logfile: General
		log = open(get_filename(eval_folder_name, 'eval_sim', 'txt'), 'w')
		log.write('Evaluation of Word Embeddings.\n'
				'Implementation by Lucas Rettenmeier, Heidelberg Institute for Theoretical Studies.\n\n')

		write_log_line(log, 'Created on', datetime.datetime.now() , 36)

		# Logfile: Evaluation Results
		log.write('{:36s} {:5d} [{:5d}]\n'.format(str('Spearman Rank:'), result[0], result[1]))
	
		# Logfile: Model Details
		log.write('\nEmbedding Model Details:\n')
		write_log_line(log, 'Embedding Model', model.model_type, 36)
		write_log_line(log, 'Number of Dimensions', model.dim_num , 36)
		write_log_line(log, 'Vocabulary Size', model.voc_size , 36)
		write_log_line(log, 'Total Word Count', model.total_count , 36)
		for key in model.training_parameters:
			write_log_line(log, key, model.training_parameters[key], 36)
		log.close()

	return result