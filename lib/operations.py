#-------------------------------------------------------------------------------------------------------------------
# Packages & Settings
#-------------------------------------------------------------------------------------------------------------------

# General Packages
import os
import datetime
import copy

# Math and Data Structures
import numpy as np
import random

# Writing and Reading from / to Disk Files
import pickle

#-------------------------------------------------------------------------------------------------------------------
# Loading own Modules
#-------------------------------------------------------------------------------------------------------------------

from lib.model import Model
from lib.util import get_filename

#---------------------------------------------------------------------------------------------------------------
# Empty copy: Create a copy of a model with empty word embeddings
#---------------------------------------------------------------------------------------------------------------

def empty_copy(model):
	# Initialize the new model as deepcopy of the original model
	new_model = copy.deepcopy(model)

	# Make it empty
	new_model.count[:] = 0
	new_model.total_count = 0
	new_model.avg[:] = 0
	new_model.embeddings[:,:] = 0

	new_model.changed = True

	return new_model

#---------------------------------------------------------------------------------------------------------------
# ALIGN VOCABULARY: Align vocabulary of two models and return aligned models, as well as an array of the type
# 	[0,1,...,m-1,m] that contains all joint indices.
#---------------------------------------------------------------------------------------------------------------

def align(model1, model2):
	# Initialize the aligned models as deepcopy of the original models
	new_model1 = copy.deepcopy(model1)
	new_model2 = copy.deepcopy(model2)

	# Create new vocabularies: A joint vocab, and two vocabularies that start with the joint words and 
	# 	then continue with unique words
	joint_voc = []
	match_indices1 = []
	match_indices2 = []
	joint_count = 0

	new_model1.indices = dict()
	new_model2.indices = dict()

	# Joint vocabulary
	for word in model1.indices:
		if word in model2.indices:

			joint_voc.append(word)

			new_model1.indices[word] = joint_count
			new_model2.indices[word] = joint_count
			
			match_indices1.append(model1.indices[word])
			match_indices2.append(model2.indices[word])

			joint_count += 1	

	# Appending Vocabulary 1 & 2
	model1_count = joint_count
	for word in model1.indices:
		if word not in new_model1.indices:
			new_model1.indices[word] = model1_count
			match_indices1.append(model1.indices[word])
			model1_count += 1

	model2_count = joint_count
	for word in model2.indices:
		if word not in new_model2.indices:
			new_model2.indices[word] = model2_count
			match_indices2.append(model2.indices[word])
			model2_count += 1

	new_model1.words = dict(map(reversed, new_model1.indices.items()))
	new_model2.words = dict(map(reversed, new_model2.indices.items()))

		# Align counts, embeddings and averages of the models
	new_model1.avg = model1.avg[match_indices1] 
	new_model1.count = model1.count[match_indices1]
	new_model1.embeddings = model1.embeddings[match_indices1]

	new_model2.avg = model2.avg[match_indices2] 
	new_model2.count = model2.count[match_indices2]
	new_model2.embeddings = model2.embeddings[match_indices2]

	return new_model1, new_model2, np.arange(len(joint_voc))

#---------------------------------------------------------------------------------------------------------------
# AVERAGING: Average the embeddings of two aligned models. 
# 	If the models don't have the same vocabulary, the array joint_indices should contain the indices 
# 	of the joint words. 
#  	The procedure is then: average embeddings over joint words, concatenate with embeddings of unique words
#---------------------------------------------------------------------------------------------------------------

def avg(model1, model2, normalize = True, joint_indices = None):
	if joint_indices is None:
		joint_indices = np.arange(model1.voc_size)

	# Number of joint words, unique words in 1 and unique words in 2
	joint_num = len(joint_indices)
	only1_num = model1.voc_size - joint_num
	only2_num = model2.voc_size - joint_num

	# Index arrays in joint model
	only1_indices = np.arange(joint_num, joint_num + only1_num)
	only2_indices = np.arange(model1.voc_size, model1.voc_size + only2_num)

	# Initialize the new model as deepcopy of the original model
	new_model = copy.deepcopy(model1)

	# Vocabulary
	for word in model2.indices:
		if word not in model1.indices:
			new_model.words[model2.indices[word] + only1_num] = word

	new_model.voc_size = len(new_model.words)
	new_model.indices = dict(map(reversed, new_model.words.items()))

	# Number of averages per word:
	new_model.avg[joint_indices] = model1.avg[joint_indices]  + model2.avg[joint_indices]
	new_model.avg = np.concatenate((new_model.avg, model2.avg[joint_num:]))

	# Word count:
	new_model.count[joint_indices] = model1.count[joint_indices] +  model2.count[joint_indices]
	new_model.count = np.concatenate((new_model.count, model2.count[joint_num:]))
	new_model.total_count = np.sum(new_model.count)

	# Average embeddings over the joint values:
	new_model.embeddings[joint_indices] = model1.embeddings[joint_indices]  + model2.embeddings[joint_indices]
	new_model.embeddings = np.concatenate((new_model.embeddings, model2.embeddings[joint_num:]), axis = 0)
	
	if (normalize):
		new_model.normalize()
	else:
		new_model.embeddings[joint_indices] /= 2

	# Get training information
	for key in new_model.training_parameters:
		new_model.training_parameters[key] = str(model1.training_parameters[key]) + '\n\t\t' + str(model2.training_parameters[key])

	new_model.changed = True

	return new_model


#---------------------------------------------------------------------------------------------------------------
# TBD
#---------------------------------------------------------------------------------------------------------------

def join(model1, model2, joint_indices = None):
	if joint_indices is None:
		joint_indices = np.arange(model1.voc_size)

	# Number of joint words, unique words in 1 and unique words in 2
	joint_num = len(joint_indices)
	only1_num = model1.voc_size - joint_num
	only2_num = model2.voc_size - joint_num

	# Index arrays in joint model
	only1_indices = np.arange(joint_num, joint_num + only1_num)
	only2_indices = np.arange(model1.voc_size, model1.voc_size + only2_num)
	new_model = copy.deepcopy(model1)

	# Vocabulary
	for word in model2.indices:
		if word not in model1.indices:
			new_model.words[model2.indices[word] + only1_num] = word

	new_model.voc_size = len(new_model.words)
	new_model.indices = dict(map(reversed, new_model.words.items()))

	# Join the two models -> randomly pick 
	pick_model2 = np.random.randint(2, size=len(joint_indices))

	# Word Counts
	new_model.count[pick_model2.nonzero()] = model2.count[pick_model2.nonzero()]
	new_model.count = np.concatenate((new_model.count, model2.count[joint_num:]))
	new_model.total_count = np.sum(new_model.count)
	
	# Embeddings
	new_model.embeddings[pick_model2.nonzero()] = model2.embeddings[pick_model2.nonzero()]
	new_model.embeddings = np.concatenate((new_model.embeddings, model2.embeddings[joint_num:]), axis = 0)

	# Normalize
	new_model.normalize()

	return new_model

#---------------------------------------------------------------------------------------------------------------
# TBD
#---------------------------------------------------------------------------------------------------------------

def avg_ĺist(model_list):

	new_model = copy.deepcopy(model_list[0])

	# Vocabulary
	for model in model_list[1:]:
		new_model.embeddings += model.embeddings
		new_model.avg += model.avg
		new_model.count += model.count

	new_model.normalize()

	return new_model

#---------------------------------------------------------------------------------------------------------------
# TBD
#---------------------------------------------------------------------------------------------------------------

def align_list(model_list):

	new_model_list = list()

	for model in model_list:
		new_model = copy.deepcopy(model)
		new_model_list.append(new_model)

	match_indices = [list() for m in range(len(new_model_list))]

	# Joint vocabulary
	word_num = 0
	for word in new_model_list[0].indices:
		for m in range(1, len(new_model_list)):
			new_model_list[m].indices[word] = word_num
			match_indices[m].append(model_list[m].indices[word])
		word_num += 1

	for m in range(1, len(new_model_list)):
		new_model_list[m].words 		= dict(map(reversed, new_model_list[m].indices.items()))
		new_model_list[m].avg 			= model_list[m].avg[match_indices[m]] 
		new_model_list[m].count 		= model_list[m].count[match_indices[m]]
		new_model_list[m].embeddings 	= model_list[m].embeddings[match_indices[m]]

	return new_model_list
	
#---------------------------------------------------------------------------------------------------------------
# TBD
#---------------------------------------------------------------------------------------------------------------

def get_common_vocab(folder_list):
	list_of_vocabs = list()
	common_vocab = list()
	
	average_vocab_size = 0

	print("\nDetermining common vocabulary of", len(folder_list), "embedding models.")

	# Load all vocabularies from the folder list
	folder_num = 0
	for load_data_folder in folder_list:
		with open(get_filename(load_data_folder, 'voc', 'pkl'), 'rb') as handle:
			words = pickle.load(handle)
			indices = dict(map(reversed, words.items()))
			average_vocab_size += len(words)
		print("Completed loading vocabulary", folder_num + 1, "of", len(folder_list), "from disk..." , end = '\r')
		folder_num += 1
		list_of_vocabs.append(indices)

	print('\nAverage vocabulary size:', average_vocab_size // len(folder_list))

	for word in list_of_vocabs[0]:
		word_in_all_vocabs = True
		for i in range(1,len(folder_list)):
			if word not in list_of_vocabs[i]:
				word_in_all_vocabs = False
				break
		if word_in_all_vocabs:
			common_vocab.append(word)

	print("Determined common vocabulary. Size:", len(common_vocab), "\n")
	return np.array(common_vocab)

#---------------------------------------------------------------------------------------------------------------
# TBD
#---------------------------------------------------------------------------------------------------------------

def k_means_clustering(model1, k):

	print('\nStarting k-means clustering. k =', k, '\n\n')

	# Initialize data structures
	cluster_array = None
	center_array = np.zeros((k,model1.dim_num))
	pr_center_array = None

	# Randomly initialize centers
	for i in range(k):
		rand_index = random.randint(0, model1.voc_size)
		center_array[i] = model1.embeddings[rand_index]

	terminated = False
	while (not terminated):

		# Cluster based on old centers
		distance_array = np.matmul(center_array, model1.embeddings.T)
		cluster_array = np.argmax(distance_array, axis = 0)

		# Calculate new centers
		pr_center_array = copy.deepcopy(center_array)	
		for i in range(k):
			print('Cluster', i + 1, ' - Size:', np.shape(np.where(cluster_array == i))[1])
			non_normalized = np.sum(model1.embeddings[np.where(cluster_array == i)], axis = 0)
			center_array[i] = non_normalized / np.sqrt(np.sum(np.matmul(non_normalized,non_normalized)))

		# Evaluate Termination condition:
		center_displacement = np.diagonal(np.matmul(center_array, pr_center_array.T))
		print('\n' 'Center displacement:', center_displacement, '\n\n')
		if( np.sum(np.square(center_displacement - 1)) ) < 1.e-6:
			terminated = True
			print('Terminated.\n')
	
	return cluster_array, center_array