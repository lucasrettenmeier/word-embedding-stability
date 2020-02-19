#-------------------------------------------------------------------------------------------------------------------
# Packages & Settings
#-------------------------------------------------------------------------------------------------------------------

# OS & General packages
import os
import datetime
import copy

# Math and data structure packages
import numpy as np
import scipy.linalg

#-------------------------------------------------------------------------------------------------------------------
# Loading own Modules
#-------------------------------------------------------------------------------------------------------------------

from lib.util import write_log_line, print_log_line, get_filename

#-------------------------------------------------------------------------------------------------------------------
# Default Parameters for the different transformation types
#-------------------------------------------------------------------------------------------------------------------

# Linear Transformation
default_linear_transformation_variant = 'SGD'
default_linear_objective = 'LSQ'
default_linear_initial_learning_rate = 0.001
default_linear_decrease_learning_rate = True
default_linear_batch_size = 512
default_linear_draw = 'shuffle'
default_linear_epochs = 16
default_linear_ridge  = 0

#-------------------------------------------------------------------------------------------------------------------
# TRANSFORMATION CLASS
#
# Generic class for a transformation between two sets of word embeddings.
#-------------------------------------------------------------------------------------------------------------------

class Transformation:

	#---------------------------------------------------------------------------------------------------------------
	# CONSTRUCTOR: Initialize a transformation by its type and directly train / load it, if wished.
	#---------------------------------------------------------------------------------------------------------------

	def __init__(self, transformation_type = 'orthogonal', train_at_init = False, model1 = None, model2 = None,
		joint = None, load_at_init = False, load_data_folder = None):
		self.transformation_type = transformation_type

		self.transformation = np.empty(0)
		self.dim_num_in = 0
		self.dim_num_out = 0

		self.changed = False

		# train_at_init model directly
		if (train_at_init == True):
			self.train(model1, model2, joint)

		if (load_at_init == True):
			self.load(load_data_folder)

	#---------------------------------------------------------------------------------------------------------------
	# TRAINING: Train a transformation from model1 to model2.
	#---------------------------------------------------------------------------------------------------------------

	def train(self, model1, model2, joint = None, 
		transformation_variant = None,
		objective = None,
		initial_learning_rate = None,
		decrease_learning_rate = None,
		batch_size = None,
		draw = None,
		epochs = None,
		ridge = None):

		self.voc_size = model1.voc_size
		self.dim_num_in = model1.dim_num
		self.dim_num_out = model2.dim_num

		# Set Changed to true to enable writing to logfile
		self.changed = True

		print('\nStarted training {0} transformation...'.format(self.transformation_type))

		if joint is None:
			joint = np.arange(model1.voc_size)

		# ORTHOGONAL
		if(self.transformation_type == 'orthogonal'):
			self.transformation, __  = scipy.linalg.orthogonal_procrustes(model1.embeddings[joint], 
				model2.embeddings[joint]) 

		# LINEAR
		if(self.transformation_type == 'linear'):
			# Set unspecified parameters to default
			if transformation_variant is None: transformation_variant = default_linear_transformation_variant
			if objective is None: objective = default_linear_objective
			if initial_learning_rate is None: initial_learning_rate = default_linear_initial_learning_rate
			if decrease_learning_rate is None: decrease_learning_rate = default_linear_decrease_learning_rate
			if batch_size is None: batch_size = default_linear_batch_size
			if draw is None: draw = default_linear_draw
			if epochs is None: epochs = default_linear_epochs
			if ridge is None: ridge = default_linear_ridge

			# Random initialization of transformation matrix
			self.transformation = np.random.rand(self.dim_num_in, self.dim_num_out) - 1

			if(transformation_variant == 'SGD'):
		
				for ep in range(epochs):
					# Init grad matrix for batch descent
					grad = np.zeros((self.dim_num_in, self.dim_num_out))
					
					# Need shuffled array for draw = shuffle
					if (draw == 'shuffle'):
						shuffle_arr = np.arange(len(joint))
						np.random.shuffle(shuffle_arr)

					# Loop over mini-batches
					max_batch_num = int(len(joint) / batch_size) + 1

					for i in range(max_batch_num):

						# Update learning_rate:
						learning_rate_factor = (ep + (i / max_batch_num)) / epochs
						learning_rate = initial_learning_rate * learning_rate_factor 

						# Get index array based on draw type 
						if (draw == 'fixed'):
							index_array = np.arange(batch_size * i, batch_size * (i+1))
						elif (draw == 'shuffle'):
							index_array = shuffle_arr[batch_size * i : batch_size * (i+1)]
						elif (draw == 'bootstrap'):
							index_array = np.random.randint(low = 0, high = len(joint), size = batch_size)

						# Get word vectors 
						x = np.transpose(model1.embeddings[index_array,:])
						z = np.transpose(model2.embeddings[index_array,:])

						# Calculate gradient based on optimization objective:
						if (objective == 'LSQ'):
							grad = np.matmul(np.matmul(self.transformation,x)-z,np.transpose(x))
						
						# Apply ridge if desired	                	
						if (ridge > 0):
							grad += batch_size * ridge * self.transformation

						# Update the transformation
						self.transformation += - grad * learning_rate

	
		# Store training variables in instance memory
		self.training_parameters = dict()

		if (self.transformation_type == 'linear'):
			self.training_parameters['Transformation Variant'] = transformation_variant
			self.training_parameters['Optimization Objective'] = objective		
			self.training_parameters['Initial Learning Rate'] = initial_learning_rate
			self.training_parameters['Decreasing Learning Rate'] = decrease_learning_rate
			self.training_parameters['Batch Size'] = batch_size
			self.training_parameters['Order of Word Vectors'] = draw
			self.training_parameters['Number of Epochs'] = epochs
			self.training_parameters['Ridge Regression Parameter'] = ridge

		# Store training variables of the input models
		self.model1_training_parameters = dict()
		self.model1_training_parameters['Embedding Model'] = model1.model_type
		self.model1_training_parameters['Vocabulary Size'] = model1.voc_size
		self.model1_training_parameters['Number of Dimensions'] = model1.dim_num
		self.model1_training_parameters['Total Word Count'] = model1.total_count
		for key in model1.training_parameters:
			self.model1_training_parameters[key] = model1.training_parameters[key]

		self.model2_training_parameters = dict()
		self.model2_training_parameters['Embedding Model'] = model2.model_type
		self.model2_training_parameters['Vocabulary Size'] = model2.voc_size
		self.model2_training_parameters['Number of Dimensions'] = model2.dim_num
		self.model2_training_parameters['Total Word Count'] = model2.total_count
		for key in model2.training_parameters:
			self.model2_training_parameters[key] = model2.training_parameters[key]

		# Print completion statement
		print('Succesfully learned transformation:')
		print_log_line('Transformation Type', self.transformation_type, 24)
		print_log_line('Input Dimension', self.dim_num_in, 24)
		print_log_line('Output Dimension', self.dim_num_out, 24)
		print('')

	#---------------------------------------------------------------------------------------------------------------
	# APPLY TRANSFORMATION: Apply the transformation to an embedding model.
	#---------------------------------------------------------------------------------------------------------------

	def apply_to(self, model):
		# Initialize the return value as deepcopy of the original model
		new_model = copy.deepcopy(model)

		# ORTHOGONAL
		if (self.transformation_type == 'orthogonal'):
			new_model.embeddings = np.matmul(model.embeddings, self.transformation)

		# LINEAR
		if(self.transformation_type == 'linear'):
			new_model.embeddings = np.matmul(model.embeddings, self.transformation.T)

		# Return model
		return new_model

	#---------------------------------------------------------------------------------------------------------------
	# LOAD FROM DISK: Load a transformation from the disk.
	#---------------------------------------------------------------------------------------------------------------

	def load(self, load_data_folder):
		load_data_folder = os.path.abspath(load_data_folder)

		# Load from numpy file
		loaded = np.load(get_filename(load_data_folder,'trafo', 'npz'))
		
		# LINEAR
		if(self.transformation_type == 'orthogonal'):
			self.transformation = loaded['transformation']
			self.dim_num_in = np.size(self.transformation, axis = 0)
			self.dim_num_out = np.size(self.transformation, axis = 1)

		# LINEAR
		if(self.transformation_type == 'linear'):
			self.transformation = loaded['transformation']
			self.dim_num_in = np.size(self.transformation, axis = 0)
			self.dim_num_out = np.size(self.transformation, axis = 1)

		# Print completion statement
		print('\nSuccesfully loaded transformation:')
		print_log_line('Transformation Type', self.transformation_type, 24)
		print_log_line('From file', get_filename(load_data_folder, 'trafo', 'npz'), 24)
		print_log_line('Input Dimension', self.dim_num_in, 24)
		print_log_line('Output Dimension', self.dim_num_out, 24)
		print('')


	#---------------------------------------------------------------------------------------------------------------
	# SAVE TO DISK: Save a transformation to the disk.
	#---------------------------------------------------------------------------------------------------------------

	def save(self, save_data_folder):
		save_data_folder = os.path.abspath(save_data_folder)

		# ORTHOGONAL
		if (self.transformation_type == 'orthogonal'):
			np.savez_compressed(get_filename(save_data_folder, 'trafo', 'npz'), transformation = self.transformation)

		# LINEAR
		if(self.transformation_type == 'linear'):
			np.savez_compressed(get_filename(save_data_folder, 'log', 'txt'), transformation = self.transformation)

		# Write to logfile
		if(self.changed):
			self.write_log(save_data_folder)

		# Print completion statement
		print('\nSuccesfully saved transformation:')
		print_log_line('Transformation Type', self.transformation_type, 24)
		print_log_line('Saved to file', get_filename(save_data_folder, 'trafo', 'npz'), 24)
		print_log_line('Log file', get_filename(save_data_folder, 'log', 'txt'), 24)
		print_log_line('Input Dimension', self.dim_num_in, 24)
		print_log_line('Output Dimension', self.dim_num_out, 24)
		print('')

	#---------------------------------------------------------------------------------------------------------------
	# LOGFILE: Write the information about the transformation and the underlying models to a logfile.
	#---------------------------------------------------------------------------------------------------------------

	def write_log(self, log_data_folder):

		# Prepare File
		log = open(get_filename(log_data_folder, 'log', 'txt'), 'w')
		log.write('Logfile for Word Embedding Transformation.\n'
				'Implementation by Lucas Rettenmeier, Heidelberg Institute for Theoretical Studies.\n\n')
		
		# Write Details to Log
		write_log_line(log, 'Created on', datetime.datetime.now() , 36)
		write_log_line(log, 'Transformation Model', self.transformation_type, 36)
		write_log_line(log, 'Input Dimension', self.dim_num_in, 36)
		write_log_line(log, 'Output Dimension', self.dim_num_out, 36)

		for key in self.training_parameters:
			write_log_line(log, key, self.training_parameters[key], 36)

		log.write('\nModel 1 - Parameters:\n')
		for key in self.model1_training_parameters:
			write_log_line(log, key, self.model1_training_parameters[key], 36)

		log.write('\nModel 2 - Parameters:\n')
		for key in self.model2_training_parameters:
			write_log_line(log, key, self.model2_training_parameters[key], 36)


		# Close Logfile
		log.close()
