import numpy as np 
import scipy.integrate as integrate
import math as m

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

params = np.array([[0.65160936, 0.64112154, 0.63828234],
 [0.01279464, 0.00980027, 0.01213082]])

for i in range(len(params[0])):
	print(i, integrate.quad(lambda x: function(x,i,params), 0, 1))