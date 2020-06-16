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

# Layout etc.
model_names = {'word2vec' : '\\textbf{word2vec} (skip-gram)', 'glove' : '\\textbf{GloVe}', 'fasttext' : '\\textbf{fastText} (skip-gram)'}
colors = ['blue', 'red', 'green']

# Data
data = np.load('analysis_results.npy')
models = ['word2vec', 'glove', 'fasttext']
data_type = ['genuine', 'random']
sizes = np.array([2 ** i for i in range(6)])

# Plot
figsize = (4.7, 9)
fig, ax = plt.subplots(nrows=len(models), ncols=1, sharex=True, sharey=False, figsize=figsize)

fig.text(0.03, 0.5, 'Fixed Effect Coefficient of Word Frequency (Solid Line)', va='center', rotation='vertical')
fig.text(0.965, 0.5, 'Variance Explained by Frequency (Dashed Line)', va='center', rotation=270)
fig.text(0.5, 0.02, 'Average Sample Size', ha='center')

for i,cell in enumerate(ax):
	
	# Get Model
	model = models[i]

	# Plot

	# Beta
	cell.plot(sizes, -data[i,0,:,0], "o-", label = "Genuine Historical Corpus", 	color = "blue",	linewidth = .5, markersize = 2)
	cell.plot(sizes, -data[i,1,:,0], "o-", label = "Randomized Corpus", 			color = "red",	linewidth = .5, markersize = 2)

	# Variance Explained
	ax2 = cell.twinx()
	ax2.plot(sizes, data[i,0,:,1], "x--", color = "blue",	linewidth = .5, markersize = 2)
	ax2.plot(sizes, data[i,1,:,1], "x--", color = "red",	linewidth = .5, markersize = 2)
	
	if model == 'fasttext':
		ax2.set_yticks([0,0.1,0.2,0.3,0.4])
		ax2.set_yticklabels(["0%", "10%", "20%", "30%", "40%"])
		cell.set_yticks([0.3,0.4,0.5,0.6, 0.7])

	if model == "word2vec":
		ax2.set_yticks([0,0.1,0.2,0.3,0.4])
		ax2.set_yticklabels(["0%", "10%", "20%", "30%", "40%"])
		cell.set_yticks([0,0.2,0.4,0.6])

	if model == 'glove':
		ax2.set_yticks([0.2,0.3,0.4,0.5,0.6,0.7])
		ax2.set_yticklabels(["20%", "30%", "40%", "50%", "60%", "70%"])
		cell.set_yticks([0.4,0.5,0.6,0.7,0.8])


	# Legend
	cell.legend()

	# X Axis
	# Ticks
	cell.set_xscale('log')
	cell.set_xticks([1,2,4,8,16,32])
	cell.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

	# Model Name
	if i != 2:
		cell.set_xlabel("Embedding Technique: " + model_names[model], labelpad = 10) 
	else:
		cell.set_xlabel("Embedding Technique: " + model_names[model], labelpad = 5) 

plt.tight_layout()
margins = {  #     vvv margin in inches
    "left"   :  0.7 / figsize[0],
    "bottom" :	0.8 / figsize[1],
    "right"  : 	1.0 - 0.6 / figsize[0],
    "top"    : 	1 - 1   / figsize[1]
}
fig.subplots_adjust(**margins)
plt.savefig("/home/rettenls/Drive/Documents/Studium/Masterthesis/Thesis/plots/semantic-change-law.pgf")
plt.show()