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

def function_2(x, i, j, params):
	result = gaussian(x,params[0][j], params[1][j])
	return result * integrate.quad(lambda y: function_2_helper(y,i,j,params),0,x)[0]

def function_2_helper(x, i, j, params):
	result = gaussian(x, params[0][i], params[1][i])
	for k in range(len(params[0])):
		if i == k or j == k:
			continue
		result *= error_func(x, params[0][k], params[1][k])
	return result

# Predict Overlap

params = np.array([[0.650,0.633,0.621,0.587,0.584,0.567,0.527,0.489],[0.010,0.011,0.009,0.015,0.011,0.011,0.011,0.009]])

relevant_word_num = len(params[0])

p_rank_1_arr = np.zeros(relevant_word_num)
for rel_index in range(relevant_word_num):
	p_rank_1 = integrate.quad(lambda x: function(x,rel_index,params),0,1)[0]
	p_rank_1_arr[rel_index] = p_rank_1
	print(rel_index, p_rank_1)


for rel_index_1 in range(relevant_word_num):
	p_rank_2 = p_rank_1_arr[rel_index_1]
	for rel_index_2 in range(relevant_word_num):
		if rel_index_2 == rel_index_1:
			continue
		p_rank_2 += integrate.quad(lambda x: function_2(x,rel_index_1, rel_index_2,params),0,1)[0]
	print(rel_index_1, p_rank_2)


