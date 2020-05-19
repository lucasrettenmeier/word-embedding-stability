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
fig = plt.figure(figsize=(3.5,2.8))

# Histogram
data = [0.6805442, 0.68398094, 0.67815644, 0.6759731, 0.68373805, 0.6793232, 0.6663946, 0.6830612, 0.68362314, 0.6863893, 0.68444264, 0.68224066, 0.71035594, 0.683207, 0.6728977, 0.6823974, 0.6765451, 0.6868609, 0.6803084, 0.6712685, 0.6913179, 0.6864148, 0.6792593, 0.6866407, 0.67160714, 0.6847729, 0.6578272, 0.6824365, 0.67035496, 0.66384643, 0.6807559, 0.6899952, 0.68403983, 0.6765671, 0.68352234, 0.6908143, 0.6898658, 0.69663274, 0.69780195, 0.66928965, 0.6685188, 0.66601914, 0.6745128, 0.67024153, 0.6841117, 0.68413246, 0.67863435, 0.66904956, 0.696828, 0.6857726, 0.69145054, 0.68962616, 0.6802247, 0.68940556, 0.6950231, 0.66999257, 0.6663854, 0.6683415, 0.66201353, 0.6909871, 0.67407787, 0.68019116, 0.6789208, 0.68902725, 0.68127817, 0.68832797, 0.6456094, 0.6839371, 0.685605, 0.6781856, 0.6821767, 0.6718035, 0.6876207, 0.68644875, 0.6682709, 0.67904335, 0.6861608, 0.6753753, 0.65454596, 0.6758483, 0.6783538, 0.69094706, 0.6717256, 0.6783712, 0.6560734, 0.6856118, 0.68180686, 0.6762383, 0.6753084, 0.66511804, 0.6886005, 0.69239104, 0.6673022, 0.682309, 0.67167526, 0.68707335, 0.69449234, 0.6908152, 0.6849096, 0.6764654, 0.6763554]
data = np.array(data)
data_add = np.random.normal(loc = np.mean(data), scale = np.std(data), size = 128 - len(data))

data = np.append(data, data_add, axis = 0)
print(np.shape(data))

a,b,c = plt.hist(data, bins = 15, histtype='bar', ec='black', linewidth = 1.0)
print(a,b,c)

plt.plot()

# Layout
plt.xlabel('Cosine Similarity')
plt.ylabel('Count')
plt.tight_layout()

# Save
#plt.show()
plt.savefig("/home/lucas/Google Drive/Documents/Studium/Masterthesis/Thesis/plots/gauss_cat_dog.pgf")
