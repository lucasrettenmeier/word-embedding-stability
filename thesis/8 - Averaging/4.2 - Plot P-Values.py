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

data = np.load(data_folder + 'merge_0002_independence_p_values.npy')

languages = ['\\textsc{Fi}', '\\textsc{Hi}', '\\textsc{Cs}', '\\textsc{Zh}', '\\textsc{Pt}', '\\textsc{Pl}', '\\textsc{En}']
models = ['\\textbf{word2vec}', '\\textbf{GloVe}', '\\textbf{fastText}']

languages = ['\\textsc{Fi}']
models = ['\\textbf{fastText}']

data = np.load(data_folder + 'merge_0002_gauss_p_values.npy')
plt.hist(data[0][0],bins = 20, histtype='bar', ec='black', linewidth = 0.3)
plt.show()
#plt.savefig("/home/lucas/Google Drive/Documents/Studium/Masterthesis/Thesis/plots/gauss_p_values.pgf")
