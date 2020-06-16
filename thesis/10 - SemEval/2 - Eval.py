#-------------------------------------------------------------------------------------------------------------------
# Packages & Settings
#-------------------------------------------------------------------------------------------------------------------

# General packages
import time
import sys
from glob import glob

# Math and data structure packages
import numpy as np
from scipy.stats import spearmanr

data_folder = '/home/rettenls/data/experiments/semeval/texts/'
exp_folder = '/home/rettenls/data/experiments/semeval/experiments/'
ans_folder = '/home/rettenls/data/experiments/semeval/golden_data/answer/task2/'
#-------------------------------------------------------------------------------------------------------------------
# Loading own Modules
#-------------------------------------------------------------------------------------------------------------------

import sys
sys.path.append("/home/rettenls/code/")

from lib.model 			import Model
from lib.trafo 			import Transformation
from lib.eval 			import print_nn_word, get_nn_list, get_cosine_similarity, get_ww_pip_norm
from lib.score 			import evaluate_analogy
from lib.operations 	import align, avg
from lib.util			import get_filename
from lib.prepare	 	import bootstrap_corpus, shuffle_corpus, concatenate_files

#-------------------------------------------------------------------------------------------------------------------
# Experiments
#-------------------------------------------------------------------------------------------------------------------

languages = ['english', 'german', 'latin', 'swedish']

for language in languages:

	answer_file_name = ans_folder + language + '.txt'
	answer_file = open(answer_file_name, 'r').readlines()
	
	answer_words = list()
	answer_scores = list()
	for line in answer_file:
		data = line.split('\t')
		answer_words.append(data[0])
		answer_scores.append(float(data[1][:-1]))


	my_answer_file_name = '/home/rettenls/data/experiments/semeval/trafo_answers/fasttext/answer/task2/' + language + '.txt'
	my_answer_file = open(my_answer_file_name, 'r').readlines()
	
	my_answer_words = list()
	my_answer_scores = list()
	for line in my_answer_file:
		data = line.split('\t')
		my_answer_words.append(data[0])
		my_answer_scores.append(float(data[1][:-1]))

	print(spearmanr(np.array(answer_scores), np.array(my_answer_scores)))
