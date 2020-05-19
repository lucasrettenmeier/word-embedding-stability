#-------------------------------------------------------------------------------------------------------------------
# Packages & Settings
#-------------------------------------------------------------------------------------------------------------------

# General
import os

# Math and Data Structure
import numpy as np
import math as m

# In- and Output
import pickle as pkl

#-------------------------------------------------------------------------------------------------------------------
# Loading own Modules
#-------------------------------------------------------------------------------------------------------------------

from lib.model import Model

#-------------------------------------------------------------------------------------------------------------------
# PRINT NEAREST NEIGHBORS OF WORD
#
# This function takes a model and a word and prints the k nearest neighbors of the word under the given model. 
#-------------------------------------------------------------------------------------------------------------------

def print_nn_word(model, word, k):
	# Get NN array for the respective word
	word_index_arr = np.array([(model.indices[word])])
	
	nn_arr, cos_val_arr = get_nn_list(model, model, word_index_arr, k)

	print('Nearest neighbors for word "{0}":'.format(word))
	for i in range(k):
		word = model.words[nn_arr[0][i]]
		print('{rank:>3d} {word:24s} {val:6.3f}'.format(rank = i+1, word = word, val = cos_val_arr[0][i]))
	print()

#-------------------------------------------------------------------------------------------------------------------
# FIND NEAREST NEIGHBORS
#
# This function takes two models and an array of word indices and returns the corresponding nearest neighbors within
# model1 for the word vectors from model2 as well as the cosine similarities.
# 
# NOTE: Make sure that the embedding spaces of the two models are aligned, i.e. the indices correspond to the same
# words (by calling align_voc_with).
#-------------------------------------------------------------------------------------------------------------------

def get_nn_list(model1, model2, word_indices, k, allow_self_neighbor = False, get_cosine_values = True):

	n = model1.voc_size
	m = len(word_indices)
	
	# Step 1: Calculate lengths of all individual vectors
	# -> Can be skipped, since vectors are always normalized
	#'''
	model1.normalize()
	model2.dim_num = len(model2.embeddings[0])
	model2.voc_size = len(model2.embeddings)
	model2.normalize()
	#'''
	# Step 2: Calculate cosine distances for all words with respect to all words in the word index array

	# Empty arrays for the nearest neighbor indices and the respective cosine distances
	nn_arr = []
	cos_val_arr = []

	max_dim = model1.max_dim_size(32, model1.voc_size)

	# Calculate scalar product of word with all other words
	for i in range(0, int(m / max_dim) + 1):
		lower = i*max_dim
		upper = min((i+1)*max_dim, m)
		
		A = model2.embeddings[word_indices[lower:upper],:]
		B = model1.embeddings[:,:]

		AB = np.matmul(A,B.T)
		
		# Set diagonal elements to -1 -> We don't want the word itself 
		# to be regarded as a nearest neighbor candidate 
		# Except the parameter 'allow_self_neighbor' is True
		if(allow_self_neighbor == False):
			AB[:,word_indices[lower:upper]] = -1
		
		nn_arr.extend(np.argpartition(-AB, range(k), axis = 1)[:,:k].tolist())

		if (get_cosine_values):
			cos_val_arr.extend([AB[i,nn_arr[i + lower]].tolist() for i in range(upper-lower)])
	if(get_cosine_values):
		return nn_arr, cos_val_arr
	else:
		return nn_arr

#-------------------------------------------------------------------------------------------------------------------
# GET COSINE SIMILARITY
#
# This function takes two models and returns the cosine similarity between the vectors of the two models for each
# word in the vocabularies. If a vector of word_indices is given, the similarity is only calculated for this sub-
# set of word vectors.
# 
# NOTE: Make sure that the embedding spaces of the two models are aligned, i.e. the indices correspond to the same
# words (by calling align_voc_with).
#-------------------------------------------------------------------------------------------------------------------

def get_cosine_similarity(model1, model2, word_indices = None):
	
	# Construct word vector arrays
	if (word_indices is None):
		array1 = model1.embeddings
		array2 = model2.embeddings
	else:
		array1 = model1.embeddings[word_indices]
		array2 = model2.embeddings[word_indices]

	# Total number of similarity values
	total_size = np.size(array1, axis = 0)
	result = np.empty(total_size, dtype = np.float32)

	# Calculate with numpy routines, break down into batches, to prevent memory overflow
	batch_size = min(model1.max_dim_size(32), total_size)
	
	# Loop over batches    
	for i in range(int(total_size / batch_size) + 1):
		
		# Get lower and upper index for batch
		lower = i * batch_size
		upper = min((i+1) * batch_size, total_size)
		
		# Build batch array
		A = array1[lower:upper,:]
		B = array2[lower:upper,:]
		
		diagonal_AB = np.diagonal((np.matmul(A,B.T)))
		
		result[lower:upper] = diagonal_AB
		
	return result

#-------------------------------------------------------------------------------------------------------------------
# GET WORD RELATEDNESS SIMILARITY FOR TWO WORDS WITHIN ONE MODEL
#
# This function takes a model and two words and returns the cosine similarity between the vectors of the two words
# in the respective model. This is a measure of the semantic relatedness of the two words.
#
#-------------------------------------------------------------------------------------------------------------------

def get_word_relatedness(model, word1, word2, warnings = False):

	try:	
		vec1 = model.embeddings[model.indices[word1]]
	except KeyError:
		if warnings:
			print('Word "{0}" not in vocabulary. Substituted by null-vector.'.format(word1))
		return 0

	try:	
		vec2 = model.embeddings[model.indices[word2]]
	except KeyError:
		if warnings:
			print('Word "{0}" not in vocabulary. Substituted by null-vector.'.format(word2))
		return 0

	return np.matmul(vec1, vec2)

#-------------------------------------------------------------------------------------------------------------------
# GET PIP NORM
#
# This function takes two models and returns the pairwise inner product loss (PIP) norm of the two embedding
# matrices (Yin & Shen - On the Dimensionality of Word Embeddings). 
#
# NOTE: To get comparable results over different vocabulary sizes, we return PIP / voc_size if reduced = True 
# (Default setting)
# 
# NOTE: Make sure that the embedding spaces of the two models are aligned, i.e. the indices correspond to the same
# words (by calling align_voc_with).
#-------------------------------------------------------------------------------------------------------------------

def get_pip_norm(model1, model2, word_indices = None, reduced = True, get_proxy = False, proxy_count = int(2.e4)):

	print('\nCalculating PIP norm.')
	
	# Construct word vector arrays
	if (word_indices is None):
		array1 = model1.embeddings
		array2 = model2.embeddings
	else:
		array1 = model1.embeddings[word_indices]
		array2 = model2.embeddings[word_indices]

	# Total number of similarity values
	total_size = np.size(array1, axis = 0)

	if (get_proxy and proxy_count < total_size):
		
		proxy_indices = np.arange(total_size)
		np.random.shuffle(proxy_indices)
		proxy_indices = proxy_indices[:proxy_count]

		array1 = array1[proxy_indices]
		array2 = array2[proxy_indices]

		total_size = proxy_count
	
	# Calculate with numpy routines, break down into batches, to prevent memory overflow
	batch_size = min(256, total_size)

	# Initialize norm
	squared_norm = 0

	array1_T = array1.T
	array2_T = array2.T

	# Loop over batches
	for i in range(int(total_size / batch_size) + 1):
		
		# Get lower and upper index for batch
		lower = i * batch_size
		upper = min((i+1) * batch_size, total_size)

		# Build batch array
		A = array1[lower:upper,:]
		B = array2[lower:upper,:]
		
		squared_norm += np.linalg.norm(np.matmul(A, array1.T) - np.matmul(B, array2.T)) ** (2)	

		# Print progress
		print("Progress: {:2.2%}".format(upper / total_size), end="\r")

	if(reduced) == True:
		norm = m.sqrt(squared_norm) / (total_size) 
	else:
		norm = m.sqrt(squared_norm) 

	print("Progress: {:2.2%} \nCompleted. Result for PIP norm: {:.3e}\n".format(1, norm))
	return norm

#-------------------------------------------------------------------------------------------------------------------
# GET WORDWISE PIP NORM
#
# TBD
#
# NOTE: Make sure that the embedding spaces of the two models are aligned, i.e. the indices correspond to the same
# words (by calling align_voc_with).
#-------------------------------------------------------------------------------------------------------------------

def get_ww_pip_norm(model1, model2, eval_indices, avg_indices = None):
	
	# Eval Arrays
	eval1 = model1.embeddings[eval_indices]
	eval2 = model2.embeddings[eval_indices]

	# Avg Arrays
	if avg_indices is not None:
		avg1 = model1.embeddings[avg_indices]
		avg2 = model2.embeddings[avg_indices]
	else:
		avg1 = model1.embeddings
		avg2 = model2.embeddings

	# Calculate with numpy routines, break down into batches, to prevent memory overflow
	avg_count = len(avg1)
	batch_size = min(256, avg_count)

	# Initialize norm
	squared_norm = np.zeros((len(eval1)))

	# Precalculate Transposed Arrays
	avg1_T = avg1.T
	avg2_T = avg2.T

	# Loop over batches
	for i in range(int(avg_count / batch_size) + 1):
		
		# Get lower and upper index for batch
		lower = i * batch_size
		upper = min((i+1) * batch_size, avg_count)

		# Build batch array
		avg1_T_batch = avg1_T[:,lower:upper]
		avg2_T_batch = avg2_T[:,lower:upper]

		squared_norm += np.sum(np.square(np.matmul(eval1, avg1_T_batch) - np.matmul(eval2, avg2_T_batch)), axis = 1)

		# Print progress
		print("Progress: {:2.2%}".format(upper / avg_count), end="\r")
	
	norm = np.sqrt(squared_norm / avg_count)

	print("Progress: {:2.2%}\n".format(upper / avg_count))
	return norm

#-------------------------------------------------------------------------------------------------------------------
# GET Common Vocabulary
#
# TBD
#
#-------------------------------------------------------------------------------------------------------------------

def get_common_vocab(model_dir_list):

	common_vocab = set()

	first_model = True
	for dir_name in model_dir_list:
		words = pkl.load(open(dir_name + '/voc.pkl', 'rb'))
		indices = dict(map(reversed, words.items()))

		if not first_model:
			
			# Find all words which are not in the vocab 
			not_found = set()
			for word in common_vocab:
				if word not in indices:
					not_found.add(word)

			# Remove them
			for word in not_found:
				common_vocab.remove(word)


		else:
			first_model = False
			for word in indices:
				common_vocab.add(word)

	return common_vocab