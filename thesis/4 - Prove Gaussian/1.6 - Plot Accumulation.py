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
fig = plt.figure(figsize=(3.6,2.6))

# Histogram
dist_folder = '/home/lucas/data/experiments/wiki/analysis/distribution/'
model = 'glove'
language = 'pl'
cos_sim = np.load(file = dist_folder + language + '_' + model + '.npy')

data = list()
for i in range(10000):
    x = cos_sim[:,i]
    data += ((x - np.mean(x)) / np.std(x)).tolist()

print(np.shape(data))

a,b,c = plt.hist(data, bins = 75, histtype='bar', ec='black', linewidth = 0.15 )
print(a,b,c)

plt.plot()

# Layout
plt.xlabel('Whitened Cosine Similarity')
plt.ylabel('Count')
plt.tight_layout()
plt.yticks([0,20000,40000,60000])

# Save
plt.show()
#plt.savefig("/home/lucas/Google Drive/Documents/Studium/Masterthesis/Thesis/plots/gauss_accumulated.pgf")
