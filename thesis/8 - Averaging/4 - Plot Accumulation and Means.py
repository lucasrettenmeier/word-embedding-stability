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

# Settings
mpl.rcParams.update(pgf_with_rc_fonts)

# Histogram
dist_folder = '/home/lucas/data/experiments/wiki/analysis/distribution/'
model = 'fasttext'
language = 'fi'

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(5, 5))

colors = ["blue", 'red']

for i,row in enumerate(ax):
	if i == 0:
		for j, cell in enumerate(row):
		    if j == 0:
		        cos_sim = np.load(file = dist_folder + language + '_' + model + '.npy')

		    if j == 1:
		        cos_sim = np.load(file = dist_folder + 'merge_normalized_0002_' + language + '_' + model + '.npy')

		    data = list()
		    for k in range(10000):
		        x = cos_sim[:,k]
		        data += (x - np.mean(x)).tolist()

		    cell.hist(data,  color = colors[j], bins = 50, histtype='bar',  ec='black', linewidth = 0.05)

		    # Layout
		    cell.set_xlabel('Cummulated Distribution')
		    cell.set_yticks([])

		    cell.set_xlim([-0.12,0.12])
		    cell.set_xticks([-0.1,0,0.1])

		    if j == 0:
		        cell.set_ylabel('Count', labelpad = 10)

	if i == 1:
		for j, cell in enumerate(row):
		    if j == 0:
		        cos_sim = np.load(file = dist_folder + language + '_' + model + '.npy')

		    if j == 1:
		        cos_sim = np.load(file = dist_folder + 'merge_normalized_0002_' + language + '_' + model + '.npy')

		    data = list()
		    for k in range(10000):
		        x = cos_sim[:,k]
		        data.append(np.mean(x))
		    cell.hist(data, color = colors[j], bins = 40, range = (-0.1,0.6), histtype='bar',  ec='black', linewidth = 0.05)

		    # Layout
		    cell.set_xlabel('Distribution of Means')
		    cell.set_yticks([])

		    if j == 0:
		        cell.set_ylabel('Count', labelpad = 10)




plt.tight_layout()
# Save
plt.savefig("/home/lucas/Google Drive/Documents/Studium/Masterthesis/Thesis/plots/average-gauss-accumulated.pgf")
plt.show()