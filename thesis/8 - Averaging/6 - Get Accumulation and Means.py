import numpy as np
import os
import numpy as np
import math
import matplotlib as mpl 

dist_folder = '/home/rettenls/data/experiments/wiki/analysis/distribution/'

languages = ['fi', 'hi', 'cs', 'zh', 'pt', 'pl', 'en']
models = ['fasttext', 'word2vec', 'glove']

for language in languages:
	for model in models:
		for size in range(1,3):
			if size == 1: 
				cos_sim = np.load(file = dist_folder + language + '_' + model + '.npy')

			if size == 2:
				cos_sim = np.load(file = dist_folder + 'merge_normalized_0002_' + language + '_' + model + '.npy')

			stds = list()
			for k in range(10000):
				x = cos_sim[:,k]
				stds.append(np.std(x))

			means = list()
			for k in range(10000):
				x = cos_sim[:,k]
				means.append(np.mean(x))
		   
			print(language, model, size, np.mean(stds), np.mean(means))