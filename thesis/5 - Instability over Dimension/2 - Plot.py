import numpy as np
import os
import pickle

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
#dist_folder = '/home/rettenls/data/experiments/wiki/analysis/word-wise-instability/'
dist_folder = '/home/lucas/data/experiments/wiki/analysis/dimension/'

data = np.array(pickle.load(open(dist_folder + 'data.pkl', 'rb'))) * 100
dims = np.array([0.1,0.2,0.5,1,2,5,10,20,50])


plt.figure(figsize=(4.5,3))
plt.errorbar(x = dims, y = data[:-1,0], yerr = data[:-1,1], fmt ='o', markersize = 3, ecolor='lightgray', elinewidth=0.7, capsize=2)

# SETTINGS
plt.xscale('log')
plt.xlabel('Vocabulary Size / Number of Dimensions')
plt.ylabel('Reduced PIP Loss $\\times 10^2$', labelpad = 5)
plt.xticks((0.1,1,10),('0.1','1','10'))

plt.tight_layout()
plt.savefig("/home/lucas/Google Drive/Documents/Studium/Masterthesis/Thesis/plots/instability-over-dimension.pgf")
plt.show()


