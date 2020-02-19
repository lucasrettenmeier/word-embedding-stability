import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats as stats


def normalize(vec):
	norm = np.sqrt(np.sum(vec * vec))
	return (vec / norm)

def cos_sim(vec1, vec2):
	return np.sum(vec1 * vec2)

dist = list()

for sample in range(int(512)):

	print('Sample num:', sample)

	for i in range(int(1.e2)):
		for j in range(i+1, int(1.e2)):

			vec1 = np.random.rand(300) - 0.5
			vec1 = normalize(vec1)

			vec2 = np.random.rand(300) - 0.5
			vec2 = normalize(vec2)

			if (i + j == 1):
				dist.append(list())

			dist[sample].append(cos_sim(vec1,vec2))
