import numpy as np
import os

import numpy as np
import math
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

data = np.load(data_folder + 'independence_p_values.npy')

languages = ['\\textsc{Fi}', '\\textsc{Hi}', '\\textsc{Cs}', '\\textsc{Zh}', '\\textsc{Pt}', '\\textsc{Pl}', '\\textsc{En}']
models = ['\\textbf{word2vec}', '\\textbf{GloVe}', '\\textbf{fastText}']

fig, ax = plt.subplots(nrows=len(languages), ncols=len(models), sharex=True, sharey=True, figsize=(5, 7))

k = 1
for i,row in enumerate(ax):
    for j,cell in enumerate(row):
        try:
            cell.hist(data[j][i],bins = 20, histtype='bar', ec='black', linewidth = 0.3)
        except:
            pass

        cell.set_xlim([0,1])
        cell.set_xticks([0,1])
        cell.set_ylim([0,500])
        cell.set_yticks([0,250])

        if i == len(ax) - 1:
            cell.set_xlabel(models[j])
        if j == 0:
            cell.set_ylabel(languages[i], labelpad = -315, rotation = 0)


plt.tight_layout()
plt.show()
plt.savefig("/home/lucas/Google Drive/Documents/Studium/Masterthesis/Thesis/plots/independence_p_values.pgf")

data = np.load(data_folder + 'gauss_p_values.npy')

fig, ax = plt.subplots(nrows=len(languages), ncols=len(models), sharex=True, sharey=True, figsize=(5, 7))

k = 1
for i,row in enumerate(ax):
    for j,cell in enumerate(row):
        try:
            cell.hist(data[j][i],bins = 20, histtype='bar', ec='black', linewidth = 0.3)
        except:
            pass

        cell.set_xlim([0,1])
        cell.set_xticks([0,1])
        cell.set_ylim([0,1050])
        cell.set_yticks([0,500])

        if i == len(ax) - 1:
            cell.set_xlabel(models[j])
        if j == 0:
            cell.set_ylabel(languages[i], labelpad = -315, rotation = 0)


plt.tight_layout()
plt.show()
plt.savefig("/home/lucas/Google Drive/Documents/Studium/Masterthesis/Thesis/plots/gauss_p_values.pgf")
