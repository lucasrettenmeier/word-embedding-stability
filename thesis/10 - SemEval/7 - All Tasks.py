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

import pickle as pkl

data_folder = '/home/rettenls/data/experiments/semeval/texts/'
exp_folder = '/home/rettenls/data/experiments/semeval/experiments/'
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
models = [sys.argv[1]]
model_types = {'word2vec': ['skipgram'], 'fasttext': ['skipgram'], 'glove': [None]}
corpora = ['corpus1', 'corpus2']
data_types = ['shuffle', 'bootstrap']
sizes = [1,2,4,8,16,32]
max_run_num = 32

results = dict()
for model in models:
	for language in languages:
		# Task 1
		ans_folder = '/home/rettenls/data/experiments/semeval/golden_data/answer/task1/'
		answer_file_name = ans_folder + language + '.txt'
		answer_file = open(answer_file_name, 'r').readlines()
		answer_words = list()
		answer_bin = list()
		for line in answer_file:
			data = line.split('\t')
			answer_words.append(data[0])
			answer_bin.append(int(data[1][:-1]))
		answer_bin = np.array(answer_bin)

		# Task 2
		ans_folder = '/home/rettenls/data/experiments/semeval/golden_data/answer/task2/'
		answer_file_name = ans_folder + language + '.txt'
		answer_file = open(answer_file_name, 'r').readlines()
		answer_scores = list()
		for line in answer_file:
			data = line.split('\t')
			answer_scores.append(float(data[1][:-1]))

		for model_type in model_types[model]:
			for data_type in data_types:
				for size in sizes:
					task1_results = list()
					task2_results = list()
					for run_number in range(max_run_num // size):

						if model_type is None:
							folder1 = exp_folder + language + '/' + corpora[0] + '/' + model + '/' + data_type
						else:
							folder1 = exp_folder + language + '/' + corpora[0] + '/' + model + '/' + model_type + '/' + data_type

						if model_type is None:
							folder2 = exp_folder + language + '/' + corpora[1] + '/' + model + '/' + data_type
						else:
							folder2 = exp_folder + language + '/' + corpora[1] + '/' + model + '/' + model_type + '/' + data_type

						if size == 1:
							run_folder1 = folder1 + '/run_{:04d}'.format(run_number)
							run_folder2 = folder2 + '/run_{:04d}'.format(run_number)
						else:
							run_folder1 = folder1 + '/merge_{:04d}_run_{:04d}'.format(size, run_number)
							run_folder2 = folder2 + '/merge_{:04d}_run_{:04d}'.format(size, run_number)
		
						m1 = Model(model)
						m1.load(run_folder1)
						m1.normalize()

						m2 = Model(model)
						m2.load(run_folder2)
						m2.normalize()

						m1,m2,joint = align(m1,m2)
						eval_indices = [m1.indices[w] for w in answer_words]
						t = Transformation('orthogonal', train_at_init = True, model1 = m1, model2 = m2, joint = joint)

						disp = 1 - get_cosine_similarity(t.apply_to(m1), m2, word_indices = joint)
						treshold = np.mean(disp) + 0.5 * np.std(disp)

						binary = np.array([int(disp[i] > treshold) for i in eval_indices])
						ranking = np.array([disp[i] for i in eval_indices])

						task1_results.append(np.mean(binary == answer_bin))
						task2_results.append(spearmanr(answer_scores, ranking)[0])

					print(model + '_' + language + '_' + data_type + '_' + str(size))
					print(task1_results)
					print(task2_results)

					results['task1_' + model + '_' + language + '_' + data_type + '_' + str(size)] = task1_results
					results['task2_' + model + '_' + language + '_' + data_type + '_' + str(size)] = task2_results

pkl.dump(results, open('all_results_' + sys.argv[1] + '.pkl', 'wb'))