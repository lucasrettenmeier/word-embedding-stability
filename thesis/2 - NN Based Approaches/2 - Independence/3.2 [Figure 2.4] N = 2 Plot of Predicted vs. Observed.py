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
data_folder = '/home/lucas/data/experiments/wiki/analysis/overlap/N=2/'

data = np.load(data_folder + 'n2_result_fi_fasttext.npy')
			
x = data[0] / 2
y = data[1] / 2

fig = plt.figure(figsize=(3.5,2.8))

# X = Y 
interval = np.linspace(-0.2,1.2,100)
plt.plot(interval, interval, color = 'red', linestyle = "dashed", markersize = 0, linewidth = 1)

# DATA
plt.scatter(x,y, s = 3)

# SETTINGS
plt.xlabel('Prediction of $p_{@2}(w_t)$')
plt.ylabel('Observation of $p_{@2}(w_t)$')
plt.tight_layout()
plt.ylim(-0.05,1.05)
plt.xlim(-0.05,1.05)

plt.savefig("/home/lucas/Google Drive/Documents/Studium/Masterthesis/Thesis/plots/prediction_observation_overlap_n=2_fi_fasttext.pgf")
plt.show()