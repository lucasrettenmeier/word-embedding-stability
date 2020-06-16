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
languages = ['hi', 'cs', 'pt']

exp_folder  = '/home/lucas/data/experiments/wiki/'

merge_levels = [1,2,4,8,16,32,64,128]

results = dict()
for model in models:
	model_results = dict()
	for language in languages:
		
		lang_results = list()

		if model == 'glove':
			folder = exp_folder + language + '/' + model + '/' + data_type + '/'
		else:
			folder = exp_folder + language + '/' + model + '/' + model_type + '/' + data_type + '/'


		for merge_size in merge_levels:

			size_results = list()
			
			if merge_size != 1:
				size_folder = folder + 'merge_nnz_{:04d}_'.format(merge_size)
			else:
				size_folder = folder

			for run_number in range(128 // merge_size):
				try:
					f = open(size_folder + 'run_{:04d}/eval.txt'.format(run_number))
					lines = f.readlines()
					size_results.append(float(lines[7][-8:-3].replace('(','')))
				except:
					pass

			size_results = np.array(size_results)
			
			mean = np.mean(size_results)
			std = np.std(size_results)

			lang_results.append(np.array((mean,std)))

		lang_results = np.array(lang_results)
		model_results[language] = lang_results / lang_results[0][0]
	results[model] = model_results

language_names = {'hi': '\\textsc{Hi}', 'fi': '\\textsc{Fi}', 'zh': '\\textsc{Zh}', 'cs': '\\textsc{Cs}', 'pt': '\\textsc{Pt}', 'pl': '\\textsc{Pl}'}
model_names = {'word2vec' : '\\textbf{word2vec}', 'glove' : '\\textbf{GloVe}', 'fasttext' : '\\textbf{fastText}'}

fig, ax = plt.subplots(nrows=len(models), ncols=1, sharex=True, sharey=False, figsize=(4.5, 7.7))

colors = ['blue', 'red', 'green']

for i,cell in enumerate(ax):
	
	model = models[i]

	lang_index = 0
	for language in languages:
		cell.errorbar(x = np.array(merge_levels), y = results[model][language][:,0], yerr = results[model][language][:,1], label = language_names[language], color = colors[lang_index], ecolor = 'lightgray', fmt = 'o-', elinewidth=.5, capsize=1.5, markeredgewidth = .5, markersize = 2, linewidth = .5)
		lang_index +=1 

	# Plot
	cell.legend()
	if model == 'fasttext':
		cell.set_yticks( [1, 1.1,1.2,1.3])
		cell.set_yticklabels( ['Base', '$+ 10\%$', '$+ 20\%$', '$+ 30\%$'])
		cell.legend(loc = 'lower center', ncol = 3)
		cell.set_ylim([0.88,1.4])
	if model == 'glove':
		cell.set_yticks( [1, 1.05,1.10,1.15])
		cell.set_yticklabels( ['Base', '$+ 5\%$', '$+ 10\%$', '$+ 15\%$'])
		cell.set_ylim([0.98,1.17])
		cell.legend(loc = 'lower center', ncol = 3)
	if model == 'word2vec':
		cell.set_yticks( [1, 1.05,1.10,1.15])
		cell.set_yticklabels( ['Base', '$+ 5\%$', '$+ 10\%$', '$+ 15\%$'])
		cell.set_ylim([0.96,1.17])
		cell.legend(loc = 'lower center', ncol = 3)
		
	cell.set_ylabel('Relative Score', rotation = 90)

	secax = cell.secondary_yaxis('right')
	secax.set_ylabel(model_names[model], rotation = 90) 
	secax.set_ticks([])

	cell.set_xticks([1,16,32,64,128])

	if (i == 2):
		cell.set_xlabel('Average Sample Size')


plt.tight_layout()
plt.savefig("/home/lucas/Google Drive/Documents/Studium/Masterthesis/Thesis/plots/score-average.pgf")
plt.show()