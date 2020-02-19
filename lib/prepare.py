#-------------------------------------------------------------------------------------------------------------------
# Packages & Settings
#-------------------------------------------------------------------------------------------------------------------

# General Packages
import os
import sys
from subprocess import call

# Math and Data Structures
import numpy as np
import random

#-------------------------------------------------------------------------------------------------------------------
# Loading own Modules
#-------------------------------------------------------------------------------------------------------------------

from lib.util import get_filename

#-------------------------------------------------------------------------------------------------------------------
# Preprocessing Methods
#-------------------------------------------------------------------------------------------------------------------

def bootstrap_corpus(file_in, file_out, ratio = 1):
	input_file = open(file_in, 'r')
	output_file = open(file_out, 'w')

	input_lines = input_file.readlines()

	count = 0
	for i in range(int(ratio * len(input_lines))):  
		j = random.randint(0, len(input_lines) -1)
		output_file.write(input_lines[j])

	input_file.close()
	output_file.close()

def shuffle_corpus(file_in, file_out, ratio = 1):
	input_file = open(file_in, 'r')
	output_file = open(file_out, 'w')

	input_lines = input_file.readlines()

	indices = np.arange(len(input_lines))
	np.random.shuffle(indices)

	count = 0

	for i in range(int(ratio * len(input_lines))):  
		j = indices[i]
		output_file.write(input_lines[j])

	input_file.close()
	output_file.close()

def concatenate_files (in_directory, concatenate_num, instance_num, total_file_num):
	
	indices = np.arange(total_file_num)
	np.random.shuffle(indices)

	instance_num = min(instance_num, total_file_num // concatenate_num)

	for instance in range(instance_num):
		output_filename = in_directory + '/cct_{:04d}_run_{:04d}.txt'.format(concatenate_num, instance)
		output_file = open(output_filename, 'w')

		for file_num in range(concatenate_num):
			index = indices[file_num + instance * concatenate_num]
			input_filename = in_directory + '/run_{:04d}.txt'.format(index)
			with open(input_filename) as input_file:
				for line in input_file:
					output_file.write(line)
