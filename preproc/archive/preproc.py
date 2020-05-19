#-------------------------------------------------------------------------------------------------------------------
# Packages & Settings
#-------------------------------------------------------------------------------------------------------------------

# General packages
import time
from glob import glob

# Math and data structure packages
import numpy as np

data_folder = '/home/rettenls/data//fast/nlp/rettenls/texts/wiki/'
#data_folder = '/home/lucas/Downloads/'

#-------------------------------------------------------------------------------------------------------------------
# Loading own Modules
#-------------------------------------------------------------------------------------------------------------------

from lib.model 			import Model
from lib.trafo 			import Transformation
from lib.eval 			import print_nn_word, get_nn_list
from lib.score 			import evaluate
from lib.operations 	import align, avg
from lib.util			import get_filename
from lib.preprocessing 	import make_corpus, bootstrap_corpus, shuffle_corpus

#-------------------------------------------------------------------------------------------------------------------
# Experiments
#-------------------------------------------------------------------------------------------------------------------

languages = ['en']

for language in languages:
	for file in glob(data_folder + language + '/*'):
		if file[-16:]=='preprocessed.txt':
			print(file)
			for i in range(256):
				folder_name = file[:file.rfind('/') + 1] 
				bootstrap_file = get_filename(folder_name + 'bootstrap', 'run' + str(i), 'txt')
				bootstrap_corpus(file, bootstrap_file)
				shuffle_file = get_filename(folder_name + 'shuffle', 'run' + str(i), 'txt')
				shuffle_corpus(file, shuffle_file)