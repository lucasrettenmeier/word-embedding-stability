import pickle as pkl
import math
import numpy as np
import matplotlib as mpl 

mpl.rcParams.update({
	"axes.titlesize" : "medium" 
	})

import matplotlib.pyplot as plt

plt.rcParams.update({
	"pgf.texsystem": "pdflatex",
	"pgf.preamble": [
		 r"\usepackage[utf8x]{inputenc}",
		 r"\usepackage[T1]{fontenc}",
		 ]
})
pgf_with_rc_fonts = {
	"font.family": "serif",
	"font.serif": [],   
	"font.size": 10,                   
}

mpl.rcParams.update(pgf_with_rc_fonts)
data_folder = '/home/lucas/data/experiments/wiki/analysis/distribution/'

model_type  = 'skipgram'
data_type   = 'shuffle'

models = ['word2vec', 'glove', 'fasttext']
languages = ['hi', 'fi', 'zh', 'cs', 'pl', 'pt']

exp_folder  = '/home/lucas/data/experiments/wiki/'

results = dict()

merge_levels = [1,2,4,8,16,32,64]

for model in models:
	model_results = dict()
	for language in languages:

		if model == 'glove':
			folder = exp_folder + language + '/' + model + '/' + data_type 
		else:
			folder = exp_folder + language + '/' + model + '/' + model_type + '/' + data_type

		conv_file = folder + '/convergence_analysis/analysis.pkl'
		conv_data = pkl.load(open(conv_file, 'rb'))

		means = list()
		stds = list()

		for merge_level in range(len(conv_data))[:-1]:
			vals = list()
			for cell in conv_data[merge_level]:
				vals += cell[1:]
			vals = np.array(vals) / 2

			if (len(vals) > 0):
				print('Language:', language)
				#print('Merge size:', int(math.pow(2, merge_level)), '; Number of pairs:', len(vals))
				print('{:.2f} +/- {:.2f}'.format(100 * np.mean(vals), 100 * np.std(vals)))
				means.append(100 * np.mean(vals))
				stds.append(100 * np.std(vals))
				print('')
				#print(vals)

		model_results[language] = np.array((means / means[0], stds / means[0]))
	results[model] = model_results


language_names = {'hi': '\\textsc{Hi}', 'fi': '\\textsc{Fi}', 'zh': '\\textsc{Zh}', 'cs': '\\textsc{Cs}', 'pt': '\\textsc{Pt}', 'pl': '\\textsc{Pl}'}
model_names = {'word2vec' : '\\textbf{word2vec}', 'glove' : '\\textbf{GloVe}', 'fasttext' : '\\textbf{fastText}'}

fig, ax = plt.subplots(nrows=len(models), ncols=1, sharex=True, sharey=True, figsize=(4, 8))

styles = ['.', 'x', '+']

for i,cell in enumerate(ax):
	
	model = models[i]

	lang_index = 0
	for language in ['hi', 'cs', 'pt']:
		cell.plot(np.array(merge_levels), results[model][language][0], styles[lang_index], label = language_names[language], ms = 4)
		lang_index +=1 
	x = np.linspace(1,64,1000)
	y = 1 / np.sqrt(x)
	cell.plot(x,y, label = 'Theoretical Limit', ls = 'dashed', lw = 1)

	# Plot
	cell.legend()

	cell.set_ylabel('Relative Red. PIP Loss', rotation = 90)

	secax = cell.secondary_yaxis('right')
	secax.set_ylabel(model_names[model], rotation = 90) 
	secax.set_ticks([])

	cell.set_xticks([1,8,16,32,64])

	if (i == 2):
		cell.set_xlabel('Average Sample Size')


plt.savefig("/home/lucas/Google Drive/Documents/Studium/Masterthesis/Thesis/plots/pip-loss-average.pgf")
plt.show()