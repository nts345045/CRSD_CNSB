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



def defined_bed(profile='UW'):
	"""
	Populate a dictionary with geometry of a sinusoidal bed's
	centerline for known experiments

	Supported: 	UW - this study
				ZI15 - Zoet & Iverson (2015)
	
	:param profile: profile name to load, defaults to 'UW'
	:type profile: str, optional
	:return: values for
		- *h*: centerline bed height [m] (2x amplitude)
		- *lbda*: bed wavelength [m]
		# - *cr*: centerline radius [m]
	:rtype: dict
	"""
	if profile.upper() == 'UW':
		# 	   Step height, Viscosity, Flow exp, Wlength. ,  Slide speed (m/a)
		out = {'hh':0.0253*2,'lbda':.31425}#,'cr': 0.2}
	
	elif profile.upper() == 'ZI15':
		out = {'hh':0.0153*2, 'lbda':.183}#, 'cr':0.4}
	else:
		raise ValueError('Supported values for profile: "UW", "ZI15"')
	return out

def defined_rheology(profile='UW'):
	"""
	Populate a dictionary with defined rheologic parameters from known experiments

	Supported: 	'UW' - this study
				'ZI15' - Zoet & Iverson (2015)

	:param profile: profile name to load, defaults to 'UW'
	:type profile: str, optional
	:return: values for:
		- *BB*: effective viscosity [Pa sec^-1/n]
		- *nn*: flow law exponent [ - ]
		# - *US*: centerline linear sliding velocity
		# 	NOTE: UW provides value 15. [m a^-1]
		# 		  ZI15 provides value numpy.linspace(2,200,100) [m a^-1]
	:rtype: dict
	"""	
	if profile.upper() == 'UW':
		out = {'BB':6.3e7, 'nn': 3}#, 'US': 15}
	elif profile.upper() == 'ZI15':
		out = {'BB':6.3e7, 'nn': 3}#, 'US': np.linspace(2,200,100)}
	else:
		raise ValueError('Supported values for profile: "UW", "ZI15"')
	return out

def defined_experiment(profile='UW'):
	params = defined_bed(profile=profile)
	params.update(defined_rheology(profile=profile))
	# params.update({'US': centerline_slip_velocity})
	return params



def lvel2avel(r_ref=.2,lvel=15.):
	"""Convert linear velocity (lvel) into angular velocity (avel)

	:param r_ref: reference radius, defaults to .2 [meters]
	:type r_ref: float, optional
	:param lvel: linear velocity value, defaults to 15 [meters per year]
	:type lvel: float, optional
	:return: angular velocity in radians per unit time
	:rtype: float
	"""	
	C_ref = 2.*np.pi*r_ref
	vv = C_ref/lvel
	return vv

def avel2lvel(c_ref,avel):
	"""Convert angular velocity (avel) into linear velocity (lvel)

	:param c_ref: reference circumference [m]
	:type c_ref: float
	:param avel: angular velocity [rad / unit time]
	:type avel: float
	:return: _description_
	:rtype: _type_
	"""	
	return c_ref*(avel/(2.*np.pi))

def rpt2lvel(c_ref, rpt):
	"""Convert rotations per unit time (rpt) into linear velocity (lvel)

	:param c_ref: reference circumference [m]
	:type c_ref: float
	:param rpt: rotations per unit time [years^-1]
	:type rpt: float
	:return: linear velocity [m a^-1]
	:rtype: float
	"""	
	return c_ref*rpt