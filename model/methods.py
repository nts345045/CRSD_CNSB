import numpy as np

def acot(x):
	"""
	Return the inverse cotangent of x
	:param x: input value(s)
	:type x: numpy.ndarray
	:return: output value
	:rtype: numpy.ndarray
	"""
	return np.pi/2 - np.arctan(x)