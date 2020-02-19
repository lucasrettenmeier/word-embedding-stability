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

def concatenate_files (in_directory):
	output_file = open(in_directory + '/concatenated.txt', 'w')

	i = 0
	for file in os.listdir(in_directory):
		if file.endswith('.txt'):
			input_file = open(in_directory + "/" + file)
			for line in input_file:
				output_file.write(line)
			input_file.close()
			if (((i + 1) % 10) == 0):
				print('Concatenated {0} documents...'.format(i + 1))
			i += 1
			if(i == 16):
				break
	output_file.close()