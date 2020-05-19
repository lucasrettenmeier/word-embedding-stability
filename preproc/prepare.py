#-------------------------------------------------------------------------------------------------------------------
# Packages & Settings
#-------------------------------------------------------------------------------------------------------------------

# General packages
import time
import sys
from glob import glob

# Math and data structure packages
import numpy as np

data_folder = '/home/rettenls/data/texts/wiki/'

#-------------------------------------------------------------------------------------------------------------------
# Loading own Modules
#-------------------------------------------------------------------------------------------------------------------

sys.path.append('/home/rettenls/code/')

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

language = 'en'

input_filename = data_folder + language + '/preproc/preproc.tok'

print('\nPreparing files for language {:s}.\nInput file: {:s}\n'.format(language, input_filename))

total_file_num = 128

for i in range(117, total_file_num):
	folder_name = data_folder + language + '/' 

	bootstrap_file = get_filename(folder_name + 'bootstrap', 'run_{:04d}'.format(i), 'txt')
	print('Preparing File:', bootstrap_file)
	bootstrap_corpus(input_filename, bootstrap_file)
	
	shuffle_file = get_filename(folder_name + 'shuffle', 'run_{:04d}'.format(i), 'txt')
	print('Preparing File:', shuffle_file)
	shuffle_corpus(input_filename, shuffle_file)


#for i in [2, 4, 8, 16, 32, 64, 128]:
#	concatenate_files(folder_name + 'bootstrap', i, 5, total_file_num)
#	concatenate_files(folder_name + 'shuffle', i, 5, total_file_num)