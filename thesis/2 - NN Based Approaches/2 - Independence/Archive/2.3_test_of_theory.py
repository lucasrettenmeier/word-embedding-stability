import numpy as np
import math as m
import scipy.integrate as integrate
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pearsonr

def gaussian(x, mu, sigma):
    return (1 / (m.sqrt(2*m.pi) * sigma)) * m.exp(- 0.5 * m.pow((x - mu) / sigma, 2.))

def error_func(x, mu, sigma):
	return 0.5 * (m.erf((x-mu)/(m.sqrt(2) * sigma)) + 1)

def function(x, i, params):
	result = gaussian(x,params[0][i], params[1][i])
	for j in range(len(params[0])):
		if j == i:
			continue
		result *= error_func(x, params[0][j], params[1][j])
	return result

# Definitions
sample_size = 128
query_words = 100


trials = 500
measured = np.zeros(trials)
predicted = np.zeros(trials)

for trial_num in range(trials):

	# Sample Gaussian Values
	gaussian_parameters = np.zeros((2, query_words))
	gaussian_parameters[0] = np.random.normal(loc = 0.4, scale = 0.1, size = query_words)
	gaussian_parameters[0] = gaussian_parameters[0,np.argsort(-gaussian_parameters[0])]
	gaussian_parameters[1] = np.random.normal(loc = 0.01, scale = 0.002, size = query_words)
	#print(gaussian_parameters[:,:5])

	# Sample Cosine Values
	cs_array = np.zeros((query_words, sample_size))
	for i in range(query_words):
		cs_array[i] = np.random.normal(loc = gaussian_parameters[0][i], scale = gaussian_parameters[1][i], size = sample_size)

	# Determine Overlap
	top = np.argmax(cs_array,axis = 0)
	overlap_num = 0
	pair_num = sample_size * (sample_size - 1) / 2
	for i in range(sample_size-1):
		overlap_num += np.count_nonzero(top[i+1:] == top[i])
	overlap = overlap_num / pair_num
	#print(overlap)


	# Predict Overlap
	# Step 1: Get Relevance
	relevance_threshold = 1.e-5
	relevance = np.zeros(query_words)
	relevance[0] = 1
	for query_index in range(1, query_words):

		A = (gaussian_parameters[0][query_index] - gaussian_parameters[0][0]) / \
			m.sqrt(2*(gaussian_parameters[1][query_index] * gaussian_parameters[1][query_index] + \
						gaussian_parameters[1][0] * gaussian_parameters[1][0])) 

		relevance[query_index] = 0.5 * (1 + m.erf(A))

	# Step 2: Get p_rank_1 for relevant words and predict overlap
	relevant_words = np.where(relevance > relevance_threshold)[0]
	params = gaussian_parameters[:,relevant_words]

	predicted_overlap = 0
	for rel_index in range(len(relevant_words)):
		p_rank_1 = integrate.quad(lambda x: function(x,rel_index,params),0,1)[0]
		#print(rel_index, p_rank_1)
		predicted_overlap += m.pow(p_rank_1,2)

	#print("Measured: {:.4f}, Predicted: {:.4f}, Threshold: {:e}, Number of Relevant Words: {:d}".format(overlap, predicted_overlap, relevance_threshold, len(relevant_words)))
	
	measured[trial_num] = overlap
	predicted[trial_num] = predicted_overlap

	if (trial_num % 10 == 0):
		print('Trial Number:', trial_num)

print(spearmanr(measured, predicted))
print(pearsonr(measured, predicted))
plt.scatter(measured, predicted)
plt.show()