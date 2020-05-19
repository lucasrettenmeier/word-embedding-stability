#-------------------------------------------------------------------------------------------------------------------
# Packages & Settings
#-------------------------------------------------------------------------------------------------------------------

# General Packages
import os
import datetime
import copy
import shutil

# Math and Data Structures
import numpy as np
import math as m
import random

# Writing and Reading from / to Disk Files
import pickle

# Languages / Encodings
import codecs

# Word Embedding Models:
import fasttext  # FastText
from subprocess import call # Call Native Models

# Maximum size for intermediate arrays
max_array_size = 1024 * 1024

#-------------------------------------------------------------------------------------------------------------------
# Loading own Modules
#-------------------------------------------------------------------------------------------------------------------

from lib.util import write_log_line, print_log_line, get_filename

#-------------------------------------------------------------------------------------------------------------------
# Default Parameters for the different models
#-------------------------------------------------------------------------------------------------------------------

# Folder Path for Temporary Files -> Make sure to have at least 10GB free memory space 
temp_folder = '/home/rettenls/data/temp'
code_folder = '/home/rettenls/code'
#temp_folder = '/hits/fast/nlp/rettenls/temp'

# Word2Vec
default_word2vec_model_variant 			= 'skipgram'
default_word2vec_dim_num 				= 300
default_word2vec_window_size 			= 5
default_word2vec_skipgram_alpha			= 0.025
default_word2vec_cbow_alpha				= 0.05
default_word2vec_epochs 				= 5
default_word2vec_neg_samp_num 			= 5
default_word2vec_min_count 				= 5
default_word2vec_subsampling_rate 		= 1.e-4
default_word2vec_thread_num 			= 32

# FastText
default_fasttext_model_variant 			= 'skipgram'
default_fasttext_dim_num 				= 300
default_fasttext_window_size 			= 5
default_fasttext_initial_learning_rate 	= 0.1
default_fasttext_epochs 				= 5
default_fasttext_neg_samp_num 			= 5
default_fasttext_min_count 				= 5
default_fasttext_subsampling_rate		= 1.e-4
default_fasttext_min_ngram				= 3
default_fasttext_max_ngram				= 6
default_fasttext_thread_num 			= 32
default_fasttext_verbose				= 2

# GloVe
default_glove_dim_num 					= 300
default_glove_window_size				= 10
default_glove_initial_learning_rate 	= 0.05
default_glove_epochs					= 100
default_glove_min_count 				= 5
default_glove_xmax 						= 100
default_glove_thread_num 				= 32
default_glove_binary					= 0
default_glove_model						= 1
default_glove_verbose					= 2
default_glove_memory					= 16

#-------------------------------------------------------------------------------------------------------------------
# EMBEDDING CLASS
#
# Generic class as a wrapper around any word embedding model.
#-------------------------------------------------------------------------------------------------------------------

class Model:

	#---------------------------------------------------------------------------------------------------------------
	# CONSTRUCTOR: Initialize a model by its type and directly train / load it, if wished.
	#---------------------------------------------------------------------------------------------------------------

	def __init__ (self, model_type, train_at_init = False, train_data_file = None, load_at_init = False, load_data_folder = None):

		# Initialize model variables
		self.model_type = model_type

		self.words = {} # Dict: Index -> Word
		self.indices = {} # Dict: Word -> Index

		self.count = np.empty(0)
		self.avg = np.empty(0)
		self.embeddings = np.empty(0)
		
		self.voc_size = 0
		self.dim_num = 0
		self.total_count = 0

		self.changed = False

		self.temp_folder = temp_folder + '/' + model_type + '_t=' + \
			datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '_rd=' + str(random.randint(1000, 9999)) 

		# Load at initialization of model:
		if (load_at_init == True):
			load_data_folder = os.path.abspath(load_data_folder)
			self.load(load_data_folder)

		if (train_at_init == True):
			train_data_file = os.path.abspath(train_data_file)
			self.train(train_data_file)

	#-------------------------------------------------------------------------------------------------------------
	# TRAINING: Train embeddings based on a training data file, using the native function of the model.
	#-------------------------------------------------------------------------------------------------------------

	def train(self, train_data_file,
		model_variant = None,
		dim_num = None,
		initial_learning_rate = None,
		epochs = None,
		window_size = None,
		neg_samp_num = None,
		min_count = None,
		xmax = None,
		max_iter = None,
		subsampling_rate = None,
		min_ngram = None,
		max_ngram = None,
		thread_num = None,
		binary = None,
		model = None,
		verbose = None,
		memory = None,
		vocab_file = None,
		):

		train_data_file = os.path.abspath(train_data_file)

		# Set Changed to true to enable writing to logfile
		self.changed = True

		print('\nStarted training {0} embeddings...\n'.format(self.model_type))

		# WORD2VEC
		if self.model_type == 'word2vec':

			# Set unspecified parameters to default
			if model_variant is None: model_variant = default_word2vec_model_variant
			if dim_num is None: dim_num = default_word2vec_dim_num
			if initial_learning_rate is None: 
				if model_variant == 'skipgram':
					initial_learning_rate = default_word2vec_skipgram_alpha
				else:
					initial_learning_rate = default_word2vec_cbow_alpha
			if epochs is None: epochs = default_word2vec_epochs
			if window_size is None: window_size = default_word2vec_window_size
			if neg_samp_num is None: neg_samp_num = default_word2vec_neg_samp_num
			if min_count is None: min_count = default_word2vec_min_count
			if subsampling_rate is None: subsampling_rate = default_word2vec_subsampling_rate
			if thread_num is None: thread_num = default_word2vec_thread_num

			if not os.path.exists(self.temp_folder):
				os.makedirs(self.temp_folder)

			word2vec_model = [code_folder + '/lib/w2v/word2vec', self.temp_folder + '/w2v_vectors', self.temp_folder + '/w2v_vocab']
			for i in range(len(word2vec_model)):
				word2vec_model[i] = os.path.abspath(word2vec_model[i])

			# Make sure files are executable
			call('chmod +x ' + word2vec_model[0], shell = True)

			# Call Word2Vec: Construct Command String
			word2vec_call = (word2vec_model[0] + 
							' -train ' + 		train_data_file +
							' -output ' + 		word2vec_model[1] +
							' -save-vocab '+	word2vec_model[2] +
							' -cbow ' +			str(int(model_variant != 'skipgram')) +
							' -size ' +			str(dim_num) +
							' -alpha ' + 		str(initial_learning_rate) +
							' -iter ' +			str(epochs) +
							' -window ' +		str(window_size) +
							' -negative ' +		str(neg_samp_num) +
							' -sample ' + 		str(subsampling_rate) +
							' -min-count ' +	str(min_count) +
							' -threads ' +		str(thread_num))
			if (vocab_file is not None):
				word2vec_call += ' -read-vocab ' + vocab_file
			print(word2vec_call)
			call(word2vec_call, shell = True)
			print('')

			# Store vectors and vocabulary as numpy
			self.store(word2vec_model)

		# FASTTEXT
		if self.model_type == 'fasttext':

			# Set unspecified parameters to default
			if model_variant is None: model_variant = default_fasttext_model_variant
			if dim_num is None: dim_num = default_fasttext_dim_num
			if initial_learning_rate is None: initial_learning_rate = default_fasttext_initial_learning_rate
			if epochs is None: epochs = default_fasttext_epochs
			if window_size is None: window_size = default_fasttext_window_size
			if neg_samp_num is None: neg_samp_num = default_fasttext_neg_samp_num
			if min_ngram is None: min_ngram = default_fasttext_min_ngram
			if max_ngram is None: max_ngram = default_fasttext_max_ngram
			if min_count is None: min_count = default_fasttext_min_count
			if subsampling_rate is None: subsampling_rate = default_fasttext_subsampling_rate
			if thread_num is None: thread_num = default_fasttext_thread_num
			if verbose is None: verbose = default_fasttext_verbose
			
			# Call fasttext training method
			fasttext_model = fasttext.train_unsupervised(
				input = train_data_file, 
				model =  model_variant,
				dim = dim_num,
				lr = initial_learning_rate, 
				epoch = epochs, 
				ws = window_size,
				neg = neg_samp_num,
				minn = min_ngram,
				maxn = max_ngram,
				minCount = min_count,
				t = subsampling_rate,
				thread = thread_num,
				verbose = verbose)

			# Store vectors and vocabulary as numpy
			self.store(fasttext_model)

		# GLOVE
		if self.model_type == 'glove':

			# Set unspecified parameters to default
			if dim_num is None: dim_num = default_glove_dim_num
			if initial_learning_rate is None: initial_learning_rate = default_glove_initial_learning_rate
			if epochs is None: epochs = default_glove_epochs
			if window_size is None: window_size = default_glove_window_size
			if min_count is None: min_count = default_glove_min_count
			if xmax is None: xmax = default_glove_xmax
			if thread_num is None: thread_num = default_glove_thread_num
			if binary is None: binary = default_glove_binary
			if model is None: model = default_glove_model
			if verbose is None: verbose = default_glove_verbose
			if memory is None: memory = default_glove_memory

			if not os.path.exists(self.temp_folder):
				os.makedirs(self.temp_folder)

			glove_model = [code_folder + '/lib/glv/build/vocab_count', code_folder + '/lib/glv/build/cooccur', 
				code_folder + '/lib/glv/build/shuffle', code_folder + '/lib/glv/build/glove',
				self.temp_folder + '/glv_vocab.txt', self.temp_folder + '/glv_cooccurrence.bin', 
				self.temp_folder + '/glv_cooccurrence_shuffle.shuf.bin', self.temp_folder + '/glv_vectors', 
				self.temp_folder + '/glv_overflow', self.temp_folder + '/glv_temp_shuffle',]

			for i in range(len(glove_model)):
				glove_model[i] = os.path.abspath(glove_model[i])
			
			# Make sure files are executable
			for i in range(4):
				call('chmod +x ' + glove_model[i], shell = True)

			# STEP 1: Build Vocabulary
			if (vocab_file is None):
				glove_vocab_call = (glove_model[0] + 
								' -verbose ' + 		str(verbose) +
								' -min-count ' +	str(min_count) +
								' < ' + 			train_data_file +
								' > ' + 			glove_model[4])
				#print(glove_vocab_call)
				call(glove_vocab_call, shell = True)
			else:
				glove_model[4] = vocab_file
			
			# STEP 2: Build Cooccurrence Matrix
			glove_cooccurrence_call = (glove_model[1] + 
							' -verbose ' + 		str(verbose) +
							' -memory ' +		str(memory) +
							' -window-size ' +	str(window_size) +
							' -vocab-file ' + 	glove_model[4] +
							' -overflow-file '+ glove_model[8] + 
							' < ' + 			train_data_file +
							' > ' + 			glove_model[5])
			#print(glove_cooccurrence_call)
			call(glove_cooccurrence_call, shell = True)
			
			# STEP 3: Shuffle Cooccurrence Matrix 
			glove_shuffle_call = (glove_model[2] + 
							' -verbose ' + 		str(verbose) +
							' -memory ' +		str(memory) +
							' -temp-file ' + 	glove_model[9] +
							' < ' + 			glove_model[5] +
							' > ' + 			glove_model[6])
			#print(glove_shuffle_call)
			call(glove_shuffle_call, shell = True)
			
			# STEP 4: Run GloVe
			glove_run_call = (glove_model[3] + 
							' -verbose ' + 		str(verbose) +
							' -threads ' + 		str(thread_num) +
							' -x-max ' + 		str(xmax) +
							' -vector-size ' +	str(dim_num) +
							' -iter ' +			str(epochs) +
							' -eta ' +			str(initial_learning_rate) +
							' -binary ' +		str(binary) + 
							' -model ' +		str(model) +
							' -vocab-file ' + 	glove_model[4] +
							' -input-file ' +	glove_model[6] +
							' -save-file ' +	glove_model[7])
			#print(glove_run_call)
			call(glove_run_call, shell = True)
			print('')

			# Store vectors and vocabulary as numpy
			self.store(glove_model)

		# Normalize embeddings!
		self.normalize()

		# Store training variables in instance memory
		self.training_parameters = dict()
		self.training_parameters['Model Variant'] = model_variant
		self.training_parameters['Training Data File'] = train_data_file
		self.training_parameters['Number of Epochs'] = epochs		
		self.training_parameters['Minimum Word Count'] = min_count
		self.training_parameters['Subsampling Threshold'] = subsampling_rate
		self.training_parameters['Number of Threads'] = thread_num

		if (self.model_type == 'word2vec' or self.model_type == 'fasttext'):
			self.training_parameters['Initial Learning Rate'] = initial_learning_rate
			self.training_parameters['Window Size'] = window_size
			self.training_parameters['Number of Negative Samples'] = neg_samp_num

		# Print completion statement
		print('\nSuccesfully trained word embeddings:')
		print_log_line('Model', self.model_type, 24)
		print_log_line('Training file', train_data_file, 24)
		print_log_line('Dimensions', self.dim_num, 24)
		print_log_line('Vocabulary size', self.voc_size, 24)
		print_log_line('Total word count', self.total_count, 24)
		print('')

	#---------------------------------------------------------------------------------------------------------------
	# NORMALIZE: Normalize embeddings, i.e. make sure each word vector has norm 1.
	#---------------------------------------------------------------------------------------------------------------

	def normalize(self):

		# Initialize empty numpy array for norm of individual vectors
		norm_sq = np.empty((self.voc_size,1), dtype = np.float32)

		max_dim = self.max_dim_size(32)

		# Norm of all vectors
		for i in range(0, int(self.voc_size / max_dim) + 1):
			# Get lower and upper index for the current batch
			lower = i * max_dim
			upper = min((i+1)*max_dim, self.voc_size)

			# Calculate norm
			A = self.embeddings[lower:upper,:]
			norm_sq[lower:upper,0] = np.diagonal((np.matmul(A,A.T)))

		# If norm is zero at any point, the embedding is zero -> can change norm to 1 without changing the vector
		norm = np.sqrt(np.where(norm_sq==0, 1, norm_sq))[:,0]

		# Normalize embeddings:
		self.embeddings /= norm[:, np.newaxis]

	#---------------------------------------------------------------------------------------------------------------
	# LOAD FROM DISK: Load word embeddings from the disk.
	#---------------------------------------------------------------------------------------------------------------

	def load(self, load_data_folder, load_option = 'model'):

		load_data_folder = os.path.abspath(load_data_folder)

		if (load_option == 'native'):
			if (self.model_type == 'fasttext'):
				fasttext_model = fasttext.load_model(load_data_folder)
				self.store(fasttext_model)

				# Normalize Data
				self.normalize()

			if (self.model_type == 'glove'):
				stream = open(load_data_folder, 'r')
				lines = stream.readlines()
				self.voc_size = len(lines)
				self.dim_num = len(lines[0].split(' ')) - 1

				self.count = np.empty((self.voc_size), np.float32)
				self.embeddings = np.empty((self.voc_size, self.dim_num), dtype = np.float32)

				for i in range(self.voc_size):
					line_split = lines[i].split(" ")				
					self.words[i] = line_split[0]
					self.count[i] = self.voc_size - i # We don't have count information -> rank as proxy 
					self.embeddings[i,:] = np.array([line_split[1:]], dtype = np.float32)
						
				# Model independent: 
				self.indices = dict(map(reversed, self.words.items()))
				self.total_count = np.sum(self.count)
				self.avg = np.ones(np.shape(self.count))

				# Normalize Data
				self.normalize()

			if (self.model_type == 'word2vec'):
				stream = open(load_data_folder, 'r')
				lines = stream.readlines()
				self.voc_size = int(lines[0].split(' ')[0])
				self.dim_num = int(lines[0].split(' ')[1])

				self.count = np.empty((self.voc_size), np.float32)
				self.embeddings = np.empty((self.voc_size, self.dim_num), dtype = np.float32)

				for i in range(self.voc_size):
					line_split = lines[i+1].split(" ")			
					self.words[i] = line_split[0]
					self.count[i] = self.voc_size - i # We don't have count information -> rank as proxy 
					self.embeddings[i,:] = np.array([line_split[1:]], dtype = np.float32)
						
				# Model independent: 
				self.indices = dict(map(reversed, self.words.items()))
				self.total_count = np.sum(self.count)
				self.avg = np.ones(np.shape(self.count))

				# Normalize Data
				self.normalize()

		if (load_option == 'model'):

			# Load dict from pkl file
			with open(get_filename(load_data_folder, 'voc', 'pkl'), 'rb') as handle:
				self.words = pickle.load(handle)

			# Obtain indices from voc:
			self.indices = dict(map(reversed, self.words.items()))

			# Load count, embeddings, and avg from numpy file
			np_data = np.load(get_filename(load_data_folder, 'data', 'npz'))
			self.count = np_data['count']
			self.embeddings = np_data['embeddings']
			self.avg = np_data['avg']

			self.voc_size = len(self.words)
			self.dim_num = np.size(self.embeddings, axis = 1)
			self.total_count = np.sum(self.count)

		# Add source file as a training parameter
		self.training_parameters = dict()
		self.training_parameters['Source File'] = load_data_folder
		
		# Add Load Option
		if (load_option == 'model'): 
			self.training_parameters['Load Option'] = "Lucas' Model"
		if (load_option == 'native'): 
			self.training_parameters['Load Option'] = "Native Model" 

		# Print completion statement
		print('\nSuccesfully loaded word embeddings:')
		print_log_line('Model', self.model_type, 24)
		print_log_line('From file', load_data_folder, 24)
		print_log_line('Dimensions', self.dim_num, 24)
		print_log_line('Vocabulary size', self.voc_size, 24)
		print_log_line('Total word count', self.total_count, 24)
		print('')

	#---------------------------------------------------------------------------------------------------------------
	# SAVE TO DISK: Save word embeddings to the disk.
	#---------------------------------------------------------------------------------------------------------------

	def save(self, save_data_folder):
		save_data_folder = os.path.abspath(save_data_folder)

		# Save voc to pickle file:
		with open(get_filename(save_data_folder, 'voc', 'pkl'), 'wb') as handle:
			pickle.dump(self.words, handle, protocol = pickle.HIGHEST_PROTOCOL)

		# Save data to numpy file
		np.savez_compressed(get_filename(save_data_folder, 'data', 'npz'), count = self.count, 
			embeddings = self.embeddings, avg = self.avg)

		# Write to logfile
		if(self.changed):
			self.write_log(save_data_folder)

		# Print completion statement
		print('\nSuccesfully saved word embeddings:')
		print_log_line('Model', self.model_type, 24)
		print_log_line('Voc file', get_filename(save_data_folder, 'voc', 'pkl'), 24)
		print_log_line('Data file', get_filename(save_data_folder, 'data', 'npz'), 24)
		print_log_line('Log file', get_filename(save_data_folder, 'log', 'txt'), 24)
		print_log_line('Dimensions', self.dim_num, 24)
		print_log_line('Vocabulary size', self.voc_size, 24)
		print_log_line('Total word count', self.total_count, 24)
		print('')


	#---------------------------------------------------------------------------------------------------------------
	# STORE EMBEDDINGS AS NUMPY: Takes the embeddings as produced from the underlying native models and stores 
	# 		them in a generic numpy format.
	#---------------------------------------------------------------------------------------------------------------

	def store(self, native_model):

		# WORD2VEC
		if(self.model_type == 'word2vec'):

			# Open vocabulary file
			voc_file_name = native_model[2]
			voc_file = open(voc_file_name, mode = 'r', encoding='utf-8', errors = 'surrogateescape')
			voc_file_lines = voc_file.readlines()

			# Open embeddings file
			vec_file_name = native_model[1]
			vec_file = open(vec_file_name, mode = 'r', encoding='utf-8', errors = 'surrogateescape')
			vec_file_lines = vec_file.readlines()
			
			# Get dimensions of the embeddings
			self.voc_size = int((vec_file_lines[0].split(" "))[0]) - 1
			self.dim_num = int((vec_file_lines[0].split(" "))[1])

			# Initialize memory
			self.count = np.empty((self.voc_size), dtype = np.float32)
			self.embeddings = np.empty((self.voc_size, self.dim_num), dtype = np.float32)

			# Remove the headers
			voc_file_lines = voc_file_lines[1:]
			vec_file_lines = vec_file_lines[2:]

			for i in range(0,self.voc_size):
				voc_file_split = voc_file_lines[i].split(" ")
				vec_file_split = vec_file_lines[i].split(" ")

				self.words[i] = voc_file_split[0]
				self.count[i] = voc_file_split[1]
				self.embeddings[i,:] = np.array([vec_file_split[1:-1]], dtype = np.float32)
				
			# Delete the temporary (native word2vec) files
			voc_file.close()
			vec_file.close()
			shutil.rmtree(self.temp_folder, ignore_errors=True)

		# FASTTEXT
		if(self.model_type == 'fasttext'):
			self.voc_size = len(native_model.words)
			self.dim_num = len(native_model[native_model.words[0]])
		
			# Initialize memory
			self.count = np.empty((self.voc_size), dtype = np.float32)
			self.embeddings = np.empty((self.voc_size, self.dim_num), dtype = np.float32)

			words, freq = native_model.get_words(include_freq=True)
			for i in range (0,self.voc_size):
				w = words[i]
				self.words[i] = w
				self.count[i] = freq[i]
				self.embeddings[i] = np.array(native_model[w])

		# GLOVE
		if(self.model_type == 'glove'):

			# Open vocabulary file & store in list
			voc_file_name = native_model[4]
			voc_file = open(voc_file_name, mode = 'r', encoding="utf-8")
			voc_file_lines = voc_file.read().splitlines()

			# Open embeddings file -> may be too large to store in list
			vec_file_name = native_model[7]
			vec_file = open(vec_file_name + '.txt', mode = 'r', encoding="utf-8")
			
			# These files can get extremely large > Read Line by Line
			line_count = len(voc_file_lines)
			for line_index in range(line_count):
				
				vec_file_line = vec_file.readline().rstrip('\n')

				if (line_index == 0):
			
					# Get dimensions of the embeddings
					self.voc_size = line_count
					self.dim_num = len(vec_file_line.split(" ")) - 1

					# Initialize memory
					self.count = np.empty((self.voc_size), np.float32)
					self.embeddings = np.empty((self.voc_size, self.dim_num), dtype = np.float32)

				# Split Lines
				voc_line_split = voc_file_lines[line_index].split(" ")
				vec_line_split = vec_file_line.split(" ")

				self.words[line_index] = voc_line_split[0]
				self.count[line_index] = voc_line_split[1]
				self.embeddings[line_index,:] = np.array([vec_line_split[1:]], dtype = np.float32)
				
			# Delete the temporary (native glove) files
			voc_file.close()
			vec_file.close()
			shutil.rmtree(self.temp_folder, ignore_errors=True)

		# Model independent: 
		self.indices = dict(map(reversed, self.words.items()))
		self.total_count = np.sum(self.count)
		self.avg = np.ones(np.shape(self.count))

	#---------------------------------------------------------------------------------------------------------------
	# REDUCE VOCAB: Reduce the vocabulary to the N most frequent words.
	#---------------------------------------------------------------------------------------------------------------

	def reduce(self, threshold):

		new_model = copy.deepcopy(self)
		
		# Only reduce if voc_size > treshhold
		if (self.voc_size > threshold):

			# Sort vocabulary if unsorted:
			if not (np.diff(self.count) <= 0).all():
				sorted_indices = np.argsort(-self.count)[:threshold]
			else:
				sorted_indices = np.arange(threshold)

			# Reduce to most frequent words above threshold
			new_model.words = {i: self.words[sorted_indices[i]] for i in range(len(sorted_indices))}
			new_model.indices = dict(map(reversed, new_model.words.items()))
			new_model.count = self.count[sorted_indices]
			new_model.embeddings = self.embeddings[sorted_indices]
			new_model.avg = self.avg[sorted_indices]

			new_model.total_count = np.sum(new_model.count)
			new_model.voc_size = threshold

		return new_model

	#---------------------------------------------------------------------------------------------------------------
	# REDUCE VOCAB: Reduce the vocabulary to a given vocabulary
	#---------------------------------------------------------------------------------------------------------------

	def reduce_to_vocab(self, vocab):

		new_model = Model(self.model_type)

		new_model.words = {i: vocab[i] for i in range(len(vocab))}
		new_model.indices = dict(map(reversed, new_model.words.items()))
		
		sorted_indices = np.array([self.indices[new_model.words[index]] for index in range(len(vocab))])
		
		new_model.count = self.count[sorted_indices]
		new_model.embeddings = self.embeddings[sorted_indices]
		new_model.avg = self.avg[sorted_indices]
		
		new_model.total_count = np.sum(new_model.count)
		new_model.voc_size = len(vocab)

		return new_model

	#---------------------------------------------------------------------------------------------------------------
	# LOGFILE: Write the information about the emmbedding model to a logfile.
	#---------------------------------------------------------------------------------------------------------------

	def write_log(self, log_data_folder):

		# Prepare File
		log = open(get_filename(log_data_folder, 'log', 'txt'), 'w')
		log.write('Logfile for Word Embeddings.\n'
				'Implementation by Lucas Rettenmeier, Heidelberg Institute for Theoretical Studies.\n\n')
		
		# Write Details to Log
		write_log_line(log, 'Created on', datetime.datetime.now() , 36)
		write_log_line(log, 'Embedding Model', self.model_type, 36)
		write_log_line(log, 'Number of Dimensions', self.dim_num , 36)
		write_log_line(log, 'Vocabulary Size', self.voc_size , 36)
		write_log_line(log, 'Total Word Count', self.total_count , 36)

		for key in self.training_parameters:
			write_log_line(log, key, self.training_parameters[key], 36)

		# Close Logfile
		log.close()

	#---------------------------------------------------------------------------------------------------------------
	# MAX SIZE: Return maximum dimensions for intermediate array:
	# 	if other_dimension is None: Return max size of quadratic array
	#---------------------------------------------------------------------------------------------------------------
	def max_dim_size(self, size_of_dtype, other_dimension = None):
		if other_dimension is None:
			return int(1024 * 32 / size_of_dtype)
		else:
			return int(max(128, 1024 * 32 / size_of_dtype * 1024 / other_dimension))