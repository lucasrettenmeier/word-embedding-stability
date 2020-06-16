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
#dist_folder = '/home/rettenls/data/experiments/wiki/analysis/word-wise-instability/'
dist_folder = '/home/lucas/data/experiments/wiki/analysis/word-wise-instability/'

language = 'pl'
models = ['word2vec', 'glove', 'fasttext']
name_models = ['\\textbf{word2vec}', '\\textbf{GloVe}', '\\textbf{fastText}']
fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(4, 7.5))

# GET X & Y Minimum and Maximum

x_min = 1
x_max = 0
y_min = 1
y_max = 0

for i,cell in enumerate(ax):

    # Get Data
    model = models[i]

    # Intrinsic Stability
    data_int = np.load(dist_folder + language + '_' + model + '_shuffle.npz')
    stab_int = np.mean(data_int['arr_0'], axis = 1)
    freq = data_int['arr_1'] / 120

    # Extrinsic Stability
    data_ext = np.load(dist_folder + language + '_' + model + '_bootstrap.npz')
    stab_ext = np.sqrt(np.square( np.mean(data_ext['arr_0'], axis = 1)) - np.square(stab_int))

    x_min = min(x_min, np.min(freq))
    x_max = max(x_max, np.max(freq))

    y_min = min(y_min, np.min(stab_int), np.min(stab_ext))
    y_max = max(y_max, np.max(stab_int), np.max(stab_ext))


freq, stab_int, stab_ext

for i,cell in enumerate(ax):

    # Get Data
    model = models[i]

    # Intrinsic Stability
    data_int = np.load(dist_folder + language + '_' + model + '_shuffle.npz')
    stab_int = np.mean(data_int['arr_0'], axis = 1)
    freq = data_int['arr_1'] / 120

    # Extrinsic Stability
    data_ext = np.load(dist_folder + language + '_' + model + '_bootstrap.npz')
    stab_ext = np.sqrt(np.square( np.mean(data_ext['arr_0'], axis = 1)) - np.square(stab_int))

    cell.scatter(freq, stab_int, s = 3, label = 'Intrinsic Instability')
    cell.scatter(freq, stab_ext, s = 3, label = 'Extrinsic Instability')

    # Settings
    cell.set_xlim([x_min,x_max])
    cell.set_xscale('log')
    cell.set_ylim([0, (int(100 * y_max) + 1) / 100])
    #cell.set_yticks([0,0.02,0.04,0.06])

    # Plot
    cell.legend()

    cell.set_ylabel('Word Instability', rotation = 90)

    secax = cell.secondary_yaxis('right')
    secax.set_ylabel(name_models[i], rotation = 90) 
    secax.set_ticks([])

    if (i == 2):
        cell.set_xlabel('Word Frequency')


"""

# SETTINGS
plt.xlabel('Word Frequency')
plt.ylabel('Stability')
plt.legend()

plt.ylim([0,0.04])
plt.xlim(smallest, largest)
plt.xscale('log')

"""

#plt.tight_layout()
plt.savefig("/home/lucas/Google Drive/Documents/Studium/Masterthesis/Thesis/plots/word-wise-instability-" + language + ".pgf")
plt.show()


"""

plt.tight_layout()
plt.savefig("/home/lucas/Google Drive/Documents/Studium/Masterthesis/Thesis/plots/gauss_p_values.pgf") """


"""

total_batch_num = 20 
total_sample_num = len(freq)

sort_indices = np.argsort(freq)

space = np.logspace(np.log10(smallest), np.log10(largest), num = total_batch_num + 1)
print(space, smallest, largest, largest < space[-1])

batch_data = list()
for batch_num in range(total_batch_num):
    lower = space[batch_num]
    upper = space[batch_num + 1]

    if batch_num == (total_batch_num - 1):
        indices = np.where(freq >= lower)
    else:
        indices =  np.where((freq < upper) & (freq >= lower))

    if len(indices[0]) > 25:
        batch_data_point = np.zeros((3, 2))
        batch_data_point[0][0] = (lower + upper) / 2
        batch_data_point[1][0] = np.mean(stab_int[indices])
        batch_data_point[1][1] = np.std(stab_int[indices])
        batch_data_point[2][0] = np.mean(stab_ext[indices])
        batch_data_point[2][1] = np.std(stab_ext[indices])
        batch_data.append(batch_data_point)

batch_data = np.array(batch_data)
#plt.fill_between(batch_data[:,0,0], batch_data[:,1,0]-batch_data[:,1,1], batch_data[:,1,0]+batch_data[:,1,1],  alpha=0.5, label = 'Intrinsic')
#plt.fill_between(batch_data[:,0,0], batch_data[:,2,0]-batch_data[:,2,1], batch_data[:,2,0]+batch_data[:,2,1],  alpha=0.5, label = 'Extrinsic')

"""