#-------------------------------------------------------------------------------------------------------------------
# Packages & Settings
#-------------------------------------------------------------------------------------------------------------------

# General packages
import time
import sys
from glob import glob

# Math and data structure packages
import numpy as np

data_folder = '/home/rettenls/data/experiments/semeval/texts/'

#-------------------------------------------------------------------------------------------------------------------
# Loading own Modules
#-------------------------------------------------------------------------------------------------------------------

import sys
sys.path.append("/home/rettenls/code/")

from lib.model 			import Model
from lib.trafo 			import Transformation
from lib.eval 			import print_nn_word, get_nn_list
from lib.score 			import evaluate_analogy
from lib.operations 	import align, avg
from lib.util			import get_filename
from lib.prepare	 	import bootstrap_corpus, shuffle_corpus, concatenate_files

#-------------------------------------------------------------------------------------------------------------------
# Experiments
#-------------------------------------------------------------------------------------------------------------------

languages = ['english', 'german', 'latin', 'swedish']
corpora = ['corpus1', 'corpus2']

for language in languages:
	for corpus in corpora:
		
		folder_name = data_folder + language + '/' + corpus + '/'
		input_filename = folder_name + 'fixed/original.txt'

		print('\nPreparing files for language {:s}.\nInput file: {:s}\n'.format(language, input_filename))

		total_file_num = 32

		for i in range(16, total_file_num):
			bootstrap_file = get_filename(folder_name + 'bootstrap', 'run_{:04d}'.format(i), 'txt')
			print('Preparing File:', bootstrap_file)
			bootstrap_corpus(input_filename, bootstrap_file)
			
			shuffle_file = get_filename(folder_name + 'shuffle', 'run_{:04d}'.format(i), 'txt')
			print('Preparing File:', shuffle_file)
			shuffle_corpus(input_filename, shuffle_file)

