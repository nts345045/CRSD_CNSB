
"""
:module: src.model.lliboutry_kamb_model
:auth: Nathan T. Stevens
:email: ntsteven@uw.edu (formerly nstevens@wisc.edu)
:org: Pacific Northwest Seismic Network / University of Wisconsin - Madison
:license: Creative Commons Attribution 4.0 (CC-BY-4.0)

:purpose:
	This module hosts methods for calculating  discretized solutions for cavity geometry and resultant
	drag using the double-valued quasi-equilibrium model in Zoet & Iverson (2015) using the analytic solutions
	of Lliboutry (1976) and Kamb (1986)

ToDos
- TODO: Add full citations for references
- TODO: cleanup extraneous methods
"""
import logging
import numpy as np
from src.model.util import acot

def lbda2k(lbda):
	"""Convert wavelength into wavenumber

	:param lbda: wavelength in meters
	:type lbda: float-like
	:return: wavenumber
	:rtype: float-like
	"""	
	return 2.*np.pi/lbda

def calc_h_bed(xvect, amp=0.0253, wavenumber=19.99423):
	"""Calculate the bed elevation profile of a sinusoidal bed
	using equation A1 in Appendix A of Stevens et al. (in prep)

	Default values from this study

	:param xvect: model gridpoints
	:type xvect: numpy.ndarray
	:param amp: bed obstacle amplitude (half-height), defaults to 0.0253 [m]
	:type amp: float-like, optional
	:param wavenumber: bed obstacle wavenumber, defaults to 19.99423 [m**-1]
	:type wavenumber: float-like
	:return: bed elevation profile
	:rtype: numpy.ndarray
	"""	
	h = amp*(np.cos(wavenumber*xvect) + 1)
	return h

def calc_g_roof(xvect, cavity_l, amp=0.0253):
	"""Calculate the cavity roof elevation profile in the absence of subsequent obstacles
	using equation A2 in Appendix A of Stevens et al. (in prep)

	Default values from this study
	
	:param xvect: model gridpoints
	:type xvect: numpy.ndarray
	:param cavity_l: cavity lengthscale parameter
	:type cavity_l: float-like
	:param amp: Bed obstacle amplitude (half height), defaults to 0.0253 [m]
	:type amp: float-like, optional
	:return: cavity roof elevation profile at points in xvect
	:rtype: numpy.ndarray
	"""	
	part1 = 0.5
	part2 = (1./np.pi)*np.arcsin((2.*xvect - cavity_l)/cavity_l)
	part3a = 2.*(2.*xvect - 1.)*np.sqrt(xvect*(cavity_l - xvect))
	part3b = np.pi*(cavity_l**2)
	g = np.real(amp*2*(part1 - part2 - (part3a/part3b)))
	return g

def calc_cavity_l(NN, USS=4.756469e-7, amp=0.0253,  BB=6.3e7, nn=3):
	"""Calculate the cacity lengthscale parameter Equation A3 in Appendix A
	of Stevens and others (in prep)

	Default values from this study

	:param amp: bed obstacle amplitude (half-height)
	:type amp: float-like
	:param USS: sliding velocity in meters per second [m sec**-1], defaults to 4.756469e-7.
	:type USS: float-like
	:param NN: effective pressure in Pascals [Pa]
	:type NN: float-like
	:param BB: _description_, defaults to 6.3e7
	:type BB: _type_, optional
	:param nn: _description_, defaults to 3
	:type nn: int, optional
	:return: _description_
	:rtype: _type_
	"""	
	part1 = (8.*USS*2.*amp)/np.pi
	part2 = (BB/NN)**nn
	cavity_l = np.sqrt(part1*part2)
	return cavity_l

def calc_N(Pw, HH, rho=910., g=9.81):
	"""Calculate the effective pressure from the ice-thickness and subglacial water pressure
	with equation A4 in Appendix A in Stevens et al. (in prep)
	
	:param Pw: subglacial water pressure in [Pa]
	:type Pw: float-like
	:param HH: ice thickness in meters [m]
	:type HH: float-like
	:param rho: glacier ice density in [kg m**-3], defaults to 910.
	:type rho: float-like, optional
	:param g: gravitational acceleration at the Earth's geoid in [m sec**-2], defaults to 9.81
	:type g: float, optional
	:return: effective pressure in [Pa]
	:rtype: float-like
	"""	
	return rho*g*HH - Pw

def calc_Tau(NN, Phi, wavenumber=19.99423, amp=0.0253):
	"""Calculate shear stress using equation A5 in Appendix A from Stevens et al (in prep)

	Default values from this study

	:param NN: Effective pressure in [Pa]
	:type NN: float-like
	:param Phi: Geometry parameter [ dimless ]
	:type Phi: float-like
	:param wavenumber: bed obstacle wavenumber in [m**-1], defaults to 19.99423
	:type wavenumber: float, optional
	:param amp: bed obstacle amplitude in [m], defaults to 0.0253
	:type amp: float, optional
	:return: shear stress in [Pa]
	:rtype: float-like
	"""	
	return (amp*wavenumber*0.5)*NN*Phi


def calc_Phi(SS, xc, wavenumber=19.99423):
	"""Calculate the bed geometry parameter using equation A6 in Appendix A
	from Stevens et al. (in prep)

	Default values from this study

	:param SS: Fractional ice-bed contact length [dimless]
	:type SS: float-like
	:param xc: ice-bed contact critical length in [m]
	:type xc: float-like
	:param wavenumber: bed obstacle wavenumber in [m**-1], defaults to 19.99423
	:type wavenumber: float, optional
	:return: bed geometry parameter
	:rtype: float-like
	"""	
	part1 = (np.pi*SS - 0.5*np.sin(2.*np.pi*SS))*np.sin(np.pi*SS - wavenumber*xc)
	part2 = np.sin(np.pi*SS) - np.pi*SS*np.cos(np.pi*SS)
	return part1/part2

def calc_xc(SS, wavenumber=19.99423):
	"""Calculate the ice-bed contact critical length using equation A7
	from Appendix A in Stevens and others (in prep)

	Default value from this study

	:param SS: Fractional ice-bed contact length [dimless]
	:type SS: float-like
	:param wavenumber: bed obstacle wave number in [m**-1], defaults to 19.99423
	:type wavenumber: float-like
	:return: ice-bed contact critical length in [m]
	:rtype: float-like
	"""	
	part1 = 1./wavenumber
	part2 = 2.*np.pi*(1. - SS) + np.sin(2.*np.pi*SS)
	part3 = np.sin(np.pi*SS) - np.pi*SS*np.cos(np.pi*SS)
	return part1*acot(part2/part3)

def calc_S(xr,xd,lbda=0.31425):
	"""Calculate the contact fraction S using equation A8 in Appendix A of
	Stevens et al. (in prep)

	Default values from this study

	:param xr: cavity reattachement point position relative to the nucleating bed obstacle crest (x=0) in [m]
	:type xr: float-like
	:param xd: cavity detachment point position relative to the nucleating bed obstacle crest (x=0) in [m]
	:type xd: float-like
	:param lbda: bed obstacle wavelength in [m], defaults to 0.31425
	:type lbda: float-like, optional
	:return: ice-bed contact length fraction
	:rtype: float-like
	"""	
	return 1. - (xr - xd)/lbda


def model_roof(NN, USS=15./(365*24*3600), lbda=0.31425, amp=0.0253, BB=6.3e7, nn=3, npts=5001):
	# Create modeling domain
	xvect = np.linspace(0,lbda,npts)
	# Convert wavelength to wavenumber
	wavenumber = lbda2k(lbda)
	# Calculate cavity lengthscale
	cavity_l = calc_cavity_l(NN=NN, USS=USS, amp=amp, BB=BB, nn=nn)
	# Calculate roof elevation profile
	g_roof = calc_g_roof(xvect=xvect, cavity_l=cavity_l, amp=amp)

	return xvect, g_roof

def model_bed(lbda=0.31425, amp=0.0253, npts=5001):
	xvect = np.linspace(0,lbda, npts)
	wavenumber = lbda2k(lbda)
	# Calculate bed elevation profile
	h_bed = calc_h_bed(xvect=xvect, amp=amp, wavenumber=wavenumber)
	return xvect, h_bed

def get_cavity_ends(xvect, g_roof, h_bed):
	idx = np.argwhere(np.diff(np.sign(g_roof - h_bed))).flatten()
	if len(idx) == 2:
		xd = xvect[idx[0]]
		xr = xvect[idx[1]]
	else:
		breakpoint()
	return xr, xd




def calc_profiles(NN,US=15.,lbda=.31425,hh=0.0253*2,BB=6.3e7,nn=3,npts=5001):
	"""Calculate the contact fraction parameter (:math:`S`) using the following system of equations
	from Appendix A in Stevens et al. (in prep) for a specified effective pressure (:math:`NN`) and
	sliding velocity (:math:`US`)
	.. math::
		(A1) 
		h(x) = \\frac{hh}{2} \\left(
					cos\\left(
							\\frac{2 \\pi x}{\\lambda}
						\\right)
					+ 1 \\right)

		(A2)
		g(x) = hh \\left(
						\\frac{1}{2} - 
						\\frac{1}{\\pi}sin^{-1} \\left(\\frac{2x - l_{c}}{l_{c}}\\right) - 
						\\frac{2(2x - 1)\\sqrt{x(l_{c} - x)}}{\\pi l_{c}^2}
				\\right)

		(A3)
		l_{c} = \\sqrt{\\frac{8 US hh}{\\pi} \\left(\\frac{BB}{NN}\\right)^{nn}}}

		(A8)
		S = 1 - \\frac{x_r - x_d}{\\lambda}
	
	with :math:`x_r` and :math: `x_d` calculated as the first and last positive-valued elements of
	g(x) - h(x) (i.e., where the cavity roof is above the bed elevation)


	:param NN: Effective pressure [Pa]
	:type NN: float
	:param US: Linear sliding velocity along the centerline, defaults to 15 [m a^-1]
	:type US: float, optional
	:param lbda: Undulatory bed obstacle wavelength at the centerline, defaults to .31425 [m]
	:type lbda: float, optional
	:param hh: Undulatory bed obstacle height (double amplitude) at the centerline, defaults to 0.0253*2 [m]
	:type hh: float, optional
	:param BB: Effective ice viscosity, defaults to 6.3e7 [Pa s^1/nn]
	:type BB: float, optional
	:param nn: Ice flowlaw exponent, defaults to 3 [ dimless ]
	:type nn: float, optional
	:param npts: number of points for discretizing the X-domain when modeling the cavity roof geometry, defaults to 5001
	:type npts: int, optional
	:returns: dictionary containing
		- **gx** (*numpy.ndarray*) -- cavity roof height vector
		- **xv** (*numpy.ndarray*) -- cavity roof position vector
		- **idetach** (*int*) -- index of the cavity detachment point in **xv**
		- **ireattach** (*int*) -- index of the cavity reattachment point in **xv**
	:rtype: dict
	"""	
	# Convert sliding velocity into m/sec
	Uss = US/(365*24*3600)
	# Discretize model domain
	xv = np.linspace(0,lbda,npts)
	# Calculate length-scale of the cavity in the absence of subsequent bumps (lc) (Equation A2 in Zoet & Iverson, 2015)
	lc = np.sqrt(8.*Uss*hh/np.pi*(BB/NN)**nn)
	# Calculate cavity roof elevation vector (g(x) - equation 4 in Kamb, 1987)
	gx = np.real(hh*(0.5 - (1./np.pi)*np.arcsin((2.*xv-lc)/lc) - (2.*(xv - lc) * np.sqrt(xv*(lc - xv))/(np.pi*lc**2))))
	# Calculate wavenumber
	kk = 2.*np.pi/lbda
	# Calculate bed amplitude
	aa = hh/2.
	# Calculate bed elevation vector (h(x) - defines a sinusoidal bed with wavelength lambda)
	# hx = (h/2.)*np.sin((2.*np.pi*xv/lbda) + np.pi/2.) + h/2.
	hx = aa*(np.cos(kk*xv) + 1.)
	# Calculate cavity space
	dd = gx - hx
	# breakpoint()
	# Get indices of cavity edges
	try:
		sa = np.max(np.argwhere(dd > 0))
		# Get first index of positive cavity volume
		sb = np.min(np.argwhere(dd > 0))
	# If model says no cavity, assign cavity start and end as same point
	except:
		sa = 0
		sb = 0
		# SS = 1.
	output = {'gx': gx, 'xv': xv, 'idetach': sb, 'ireattach': sa}
	return output

def calc_S(x_detach, x_reattach, lbda=0.31425):
	"""Calculate the contact fraction from cavity and bed geometry parameters

	Equation A8 in Stevens et al. (in prep)

	:param x_detach: cavity detachment point position [m]
	:type x_detach: float
	:param x_reattach: cavity reattachment point poistion [m]
	:type x_reattach: float
	:param lbda: bed obstacle wavelength, defaults to 0.31425 [m]
	:type lbda: float, optional
	:return: ice-bed contact fraction [dimless]
	:rtype: float
	"""	
	# Calculate bed contact fraction
	SS = 1. - ((x_reattach - x_detach)/lbda)
	return SS


def calc_cavity_stats(gx, xv, idetach, ireattach, hh=0.0253*2, lbda=0.31425):
	"""Calculate additional geometric (sub)parameters used in Stevens et al. (in prep)
    from outputs of :meth:`~src.model.lliboutry_kamb_model.calc_profiles`

	:param gx: cavity roof elevation profile
	:type gx: numpy.ndarray
	:param xv: cavity roof position profile
	:type xv: numpy.ndarray
	:param idetach: detachment point index
	:type idetach: int
	:param ireattach: reattachment point index
	:type ireattach: int
	:param hh: bed obstacle height, defaults to 0.0253*2 [m]
	:type hh: float, optional
	:param lbda: bed obstacle wavelength, defaults to 0.31425 [m]
	:type lbda: float, optional
	:return: dictionary containing:
		- **S** (*float*) -- ice-bed contact fraction
		- **Slee** (*float*) -- ice-bed contact fraction on the lee of the obstacle
		- **Sstoss** (*float*) -- ice-bed contact fraction on the stoss of the obstacle
		- **Rmea** (*float*) -- average fractional cavity height
		- **Rlee** (*float*) -- fractional cavity detachment point elevation
		- **Rstoss** (*float*) -- fractional cavity reattachment point elevation
	:rtype: dict
	"""	
	x_detach= xv[idetach]
	x_reattach = xv[ireattach]
	SS = calc_S(x_detach=x_detach, x_reattach=x_reattach, lbda=lbda)
	# Calculate cavity statistics
	if idetach != ireattach:
		Rmea = np.mean(gx[idetach:ireattach])/hh
		Rstoss = np.min(gx[idetach:ireattach])/hh
		Rlee = np.max(gx[idetach:ireattach])/hh
	else:
		Rmea = 0
		Rstoss = 0
		Rlee = 0
	Slee = xv[idetach]/lbda
	Sstoss = (lbda - xv[ireattach])/lbda
	output = dict(zip(['S','Slee','Sstoss','Rmea','Rlea','Rstoss'],[SS, Slee, Sstoss, Rmea, Rlee, Rstoss]))
	return output

def calc_shear_and_geoparam(NN,SS,cc=0.5, hh=0.0253*2,lbda=.31425):
	"""Calculate shear stress (:math:`\\tau`) and the geometric parameter (:math:`\\Phi`)
	from the Lliboutry/Kamb analytic solution using specified bed geometry, pre-calculated
	cavity geometry (:math:`S`) using the system of equations from Appendix A in Stevens et al. (in prep)

	.. math::
		(A5)
		\\tau = cc\\frac{hh \\pi}{\\lambda} NN \\Phi

		(A6)
		\\Phi = \\frac
			{[\\pi S - \\frac{1}{2} sin(2 \\pi S)] sin(\\pi S - \\frac{2 \\pi}{\\lambda} x^\\prime)}
			{sin(\\pi S) - \\pi S cos(\\pi S)}
		
		(A7)
		x^\\prime = \\frac{\\lambda}{2\\pi} cot^{-1} \\left(
			\\frac{2 \\pi (1 - S) + sin(2 \\pi S)}{sin(\\pi S) - \\pi S cos(\\pi S)}
		\\right)

	:param NN: Effective pressure [Pa]
	:type NN: float or numpy.ndarray
	:param SS: Bed contact fraction [ dimless ]
	:type SS: float or numpy.ndarray
	:param cc: prefactor correction from Zoet & Iverson (2015), defaults to 0.5
	:type cc: float, optional
	:param hh: Bed obstacle height (double amplitude), defaults to 0.0253*2 [m]
	:type hh: float, optional
	:param lbda: Bed obstacle wavelength, defaults to .31425 [m]
	:type lbda: float, optional
	:returns:
		- **tau** - (*float* or *numpy.ndarray*) -- shear stress [Pa]
		- **phi** - (*float* or *numpy.ndarray*) -- geometric parameter [ dimless ]
	:rtype: _type_
	"""
	if isinstance(NN, float) and isinstance(SS, float):
		pass
	elif isinstance(NN, np.ndarray) and isinstance(SS, np.ndarray):
		if NN.shape == SS.shape:
			pass
		else:
			raise ValueError
	else:
		raise TypeError
	
	kk = 2.*np.pi/lbda
	# Calculate bed amplitude
	aa = hh*cc
	# Calculate wave-number scaled cavity critical length (k*x_c)
	kxc = acot((2.*np.pi*(1. - SS) + np.sin(2.*np.pi*SS))/(1. - np.cos(2.*np.pi*SS)))
	# Calculate fo factor (Coefficient of friction)
	# fo = ((h/2.)/lbda)*np.pi*N
	fo = aa*kk*0.5*NN
	# Calculate geometric factor
	phi = ((np.pi*SS - 0.5*np.sin(2.*np.pi*SS))*np.sin(np.pi*SS - kxc))/(np.sin(np.pi*SS) - np.pi*SS*np.cos(np.pi*SS))
	# Calculate shear stress
	tau  = fo*phi
	return tau, phi


def calc_TS_single(NN, US, hh=0.0253*2, lbda=0.31425, BB=6.3e7, nn=3, npts=5001):
	"""Calculate a single parameter vector for a specified bed geometry, ice rheology,
	effective pressure, and sliding speed

	:param NN: effective pressure [Pa]
	:type NN: float
	:param US: sliding velocity [m a^-1]
	:type US: float
	:param hh: bed obstacle height, defaults to 0.0253*2 [m]
	:type hh: float, optional
	:param lbda: bed obstacle wavelength, defaults to 0.31425 [m]
	:type lbda: float, optional
	:param BB: ice effective viscosity, defaults to 6.3e7 [Pa sec^1/nn]
	:type BB: float, optional
	:param nn: ice flow law exponent, defaults to 3 [dimless]
	:type nn: float, optional
	:param npts: domain discretization number of nodes, defaults to 5001
	:type npts: int, optional
	:return: array containing
		- **NN** (*float*) -- effective pressure (input value) [Pa]
		- **US** (*float*) -- sliding velocity (input value) [m a^-1]
		- **SS** (*float*) -- ice-bed contact fraction [dimless]
		- **TT** (*float*) -- shear stress [Pa]
		- **PP** (*float*) -- geometric parameter :math:`\\Phi`
		- **RR** (*float*) -- average fractional cavity height [dimless]
	:rtype: float
	"""	
	profiles = calc_profiles(NN=NN, US=US, lbda=lbda, hh=hh, BB=BB, nn=nn, npts=npts)
	profiles.update({'hh':hh, 'lbda': lbda})
	stats = calc_cavity_stats(**profiles)
	SS = stats['S']
	RR = stats['Rmea']
	TT, PP = calc_shear_and_geoparam(NN, SS, cc=0.5, hh=hh, lbda=lbda)
	return np.array([NN, US, SS, TT, PP, RR])


def calc_SR_single(NN, US, hh=0.0253*2, lbda=0.31424, BB=6.3e7, nn=3, npts=5001):
	profiles = calc_profiles(NN=NN, US=US, lbda=lbda, hh=hh, BB=BB, nn=nn, npts=npts)
	profiles.update({'hh':hh, 'lbda': lbda})
	stats = calc_cavity_stats(**profiles)
	return np.array(list(stats.values()))

def calc_parameter_space_from_NU(Nv, Uv, hh=0.0253*2, lbda=0.31425, BB=6.3e7, nn=3, npts=5001):
	"""Wrapper function to calculate the parameter space:
	.. math::
		[S, \\tau, \\Phi, R] = f(N, U, {B, n, \\lambda, h})

	:param Nv: effective pressure vector
	:type Nv: numpy.ndarray
	:param Uv: sliding velocity vector
	:type Uv: numpy.ndarray
	:param hh: bed obstacle height, defaults to 0.0253*2
	:type hh: float, optional
	:param lbda: bed obstacle wavelength, defaults to 0.31425
	:type lbda: float, optional
	:param BB: ice effective viscosity, defaults to 6.3e7
	:type BB: float, optional
	:param nn: ice flowlaw exponent, defaults to 3
	:type nn: float, optional
	:param npts: cavity modeling discretization (number of nodes), defaults to 5001
	:type npts: int, optional
	:return: output array with shape (len(Nv), len(Uv), 6)
		with the axes corresponding to
		 - 0: Nv values
		 - 1: Uv values
		 - 2: Parameter values N, U, S, \tau, \Phi, R fields

	:rtype: numpy.ndarray
	"""	
	output_array = np.full(shape=(len(Nv), len(Uv), 6), fill_value=np.nan)
	for ii, N_ in enumerate(Nv):
		print(f'N iteration {ii+1}/{len(Nv)}')
		for jj, U_ in enumerate(Uv):
			output_array[ii,jj,:] = calc_TS_single(N_, U_, hh=hh, lbda=lbda, BB=BB, nn=nn, npts=npts)
	return output_array


def calc_geometry_space_from_NU(Nv, Uv, hh=0.0253*2, lbda=0.31425, BB=6.3e7, nn=3, npts=5001):
	output_array = np.full(shape=(len(Nv), len(Uv), 6), fill_value=np.nan)
	for ii, N_ in enumerate(Nv):
		print(f'N iteration {ii+1}/{len(Nv)}')
		for jj, U_ in enumerate(Uv):
			ijout = calc_SR_single(N_, U_, hh=hh, lbda=lbda, BB=BB, nn=nn, npts=npts)
			output_array[ii,jj,:] = ijout
	return output_array


# def bedmodel(lbda=.3*np.pi*2.*0.25,hh=.078,npts=5001,pi_offset=0.5,lbda_offset=0.5,ncycles=1.5):
# 	xv = np.linspace(-ncycles*0.5*lbda,ncycles*0.5*lbda,npts)
# 	# Calculate wavenumber
# 	kk = 2.*np.pi/lbda
# 	# Calculate bed amplitude
# 	aa = hh/2.
# 	# Calculate bed elevation vector (h(x) - defines a sinusoidal bed with wavelength lambda)
# 	# hx = (h/2.)*np.sin((2.*np.pi*xv/lbda) + np.pi/2.) + h/2.
# 	hx = aa*(np.cos(kk*xv - pi_offset*np.pi) + 1.)
# 	# Calculate get normlized locations
# 	iv = xv/lbda
# 	do = {'x_m':xv - lbda_offset*lbda,'y_m':hx,'x_ind':iv}
# 	return do




# def bedslice(r_ref):
# 	ao = 0.04; ai = 0.01375
# 	ro = 0.3;  ri = 0.1
# 	kk = 4
# 	mm = (ao - ai)/(ro - ri)
# 	a_ref = ai + (r_ref-ri)*mm 
# 	l_ref = 2.*np.pi*r_ref/kk
# 	return a_ref,l_ref








# def calc_drag(NN,hh=0.0253*2,BB=6.3e7,nn=3,lbda=0.31425,US=15,npts=5001):
# 	SS = calc_SS(NN,BB=BB,nn=nn,lbda=lbda,US=US,npts=npts)
# 	TT = calc_from_SN(NN,SS,hh=hh,lbda=lbda)
# 	return SS,TT


# def bedslice_fit(NN,nvals=31,**kwargs):
# 	r_vect = np.linspace(0.1,0.3,nvals)
# 	U_vect = avel2lvel(2.*np.pi*r_vect,lvel2avel())
# 	SSo = []; TTo = []; lbdo = []; aao = []
# 	for i_,r_ in enumerate(r_vect):
# 		a_i,l_i = bedslice(r_)
# 		S_i,T_i = calc_drag(NN,hh=a_i*2,lbda=l_i,US=U_vect[i_],**kwargs)
# 		SSo.append(S_i)
# 		TTo.append(T_i)
# 		lbdo.append(l_i)
# 		aao.append(a_i)
# 	return {'r':r_vect,'Us':U_vect,'S':SSo,'T':TTo,'lbda':lbdo,'a':aao}





# def model_TS_single(N,h=0.0253*2,B=6.3e7,nn=3,lbda=.31425,US=15,npts=5001,output='tau'):
# 	"""
# 	Model cavity geometry (S) from specified sinusoidal bed
# 	geometry (assuming 1-D obstacles) based on theory from Lliboutry (1968)

# 	:: INPUTS ::
# 	:type N: float
# 	:param N: effective stress in [Pa]
# 	:type h: float
# 	:param h: obstacle height in [m] (double step amplitude)
# 	:type B: float
# 	:param B: effective viscosity of ice in [Pa a^1/nn]
# 	:type nn: float
# 	:param nn: flowlaw exponent associated with dimensionality of B, 
# 			   from Glen (1955)
# 	:type lbda: float
# 	:param lbda: bed wavelength in [m]
# 	:type US: float
# 	:param US: sliding velocity in [m/a]
# 	:type npts: int
# 	:param npts: number of points to discretize x-domain with, default is 1000
# 	:type output: str
# 	:param output: output format
# 					'tau': (default) just output estimate of shear stress (TAU)
# 					'params': TAU, SS
# 					'model': TAU, gx, hx
# 					'full': TAU, SS, gx, hx
# 	:: RETURN ::
# 	:rtype TAU: float
# 	:
# 	:rtype SS: float
# 	:return SS: ice-bed contact fraction [length / wavelength]
# 	"""
# 	# Convert sliding velocity into m/sec
# 	Uss = US/(365*24*3600)
# 	# Discretize model domain
# 	xv = np.arange(0,lbda,1/int(npts))
# 	# Calculate length-scale of the cavity in the absence of subsequent bumps (lc) (Equation A2 in Zoet & Iverson, 2015)
# 	lc = np.sqrt(8.*Uss*h/np.pi*(B/N)**nn)
# 	# Calculate cavity roof elevation vector (g(x) - equation 4 in Kamb, 1987)
# 	try:
# 		gx = np.real(h*(0.5 - (1./np.pi)*np.arcsin((2.*xv-lc)/lc) - (2.*(xv - lc) * np.sqrt(xv*(lc - xv))/(np.pi*lc**2))))
# 	except:
# 		breakpoint()
# 	# Calculate wavenumber
# 	kk = 2.*np.pi/lbda
# 	# Calculate bed amplitude
# 	aa = h/2.
# 	# Calculate bed elevation vector (h(x) - defines a sinusoidal bed with wavelength lambda)
# 	# hx = (h/2.)*np.sin((2.*np.pi*xv/lbda) + np.pi/2.) + h/2.
# 	hx = aa*(np.cos(kk*xv) + 1.)
# 	# Calculate cavity space
# 	dd = gx - hx
# 	# breakpoint()
# 	# Get indices of cavity edges
# 	try:
# 		sa = np.max(np.argwhere(dd > 0))
# 		# Get first index of positive cavity volume
# 		sb = np.min(np.argwhere(dd > 0))
# 	# If model says no cavity, assign cavity start and end as same point
# 	except:
# 		sa = 0
# 		sb = 0
# 		# SS = 1.
# 	# Calculate bed contact fraction
# 	SS = 1. - ((xv[sa] - xv[sb])/lbda)
# 	# Calculate wave-number scaled cavity critical length (k*x_c)
# 	kxc = acot((2.*np.pi*(1. - SS) + np.sin(2.*np.pi*SS))/(1. - np.cos(2.*np.pi*SS)))
# 	# Calculate fo factor (Coefficient of friction)
# 	# fo = ((h/2.)/lbda)*np.pi*N
# 	fo = aa*kk*0.5*N
# 	# Calculate geometric factor
# 	PHI = ((np.pi*SS - 0.5*np.sin(2.*np.pi*SS))*np.sin(np.pi*SS - kxc))/(np.sin(np.pi*SS) - np.pi*SS*np.cos(np.pi*SS))
# 	# Calculate drag
# 	TAU = fo*PHI


# 	if output == 'tau':
# 		return TAU
# 	elif output == 'model':
# 		return TAU, gx, hx
# 	elif output == 'full':
# 		return TAU, SS, gx, hx
# 	elif output == 'params':
# 		return TAU, SS
# 	else:
# 		return TAU


# def calc_TS_vectors(N_min,N_max,nnods=5001,**kwargs):
# 	"""
# 	Model cavity geometry (S) and shear stress (T) for a specified 
# 	sinusoidal bed geometry across a range of effective pressures (N)
# 	based on theory from Lliboutry (1968)

# 	:: INPUTS ::
# 	:type N_min: float
# 	:param N_min: minimum effective stress in [Pa] to assess
# 	:type N_max: float
# 	:param N_max: maximum effective stress in [Pa] to assess
# 	:type nnods: int
# 	:param nnods: number of points to discretize N-domain , default is 5001

# 	kwargs: see model_TS_single()

# 	:: OUTPUT ::
# 	:return: [dict] Modeled value vectors
# 					'T_Pa' - shear stress in Pa
# 					'S' - contact area fraction [fract]
# 					'N_Pa' - effective stress in Pa
# 	"""

# 	TV = []; SV = []
# 	for i_,N_ in enumerate(np.linspace(N_min,N_max,nnods)):
# 		iT,iS = model_TS_single(N_,output='params',npts=nnods,**kwargs)
# 		TV.append(iT)
# 		SV.append(iS)
# 	return {'T_Pa':np.array(TV),'S':np.array(SV),'N_Pa':np.linspace(N_min,N_max,nnods)}


# def calc_SR_vectors(N_min,N_max,nnods=5001,q=0.125,**kwargs):
# 	Rstoss, Rmea, Rlee, Sstoss, Stot, Slee = [],[],[],[],[],[]
# 	for N_ in np.linspace(N_min,N_max,nnods):
# 		outs_ = calc_SS(N_,**kwargs,q=q,output='stats')
# 		Stot.append(outs_[0])
# 		Slee.append(outs_[1])
# 		Sstoss.append(outs_[2])
# 		Rmea.append(outs_[3])
# 		Rlee.append(outs_[4])
# 		Rstoss.append(outs_[5])
# 	OUTS = {'Stot':Stot,'Slee':Slee,'Sstoss':Sstoss,\
# 			'Rmea':Rmea,'Rlee':Rlee,'Rstoss':Rstoss}
# 	return OUTS





# # CODE VALIDATION SECTION

# def run_ISU_test2(Uval):
# 	CLR = .4
# 	CLC = 2.*np.pi*CLR
# 	CLW = .183
# 	CLa = .0153
# 	BB = 6.3e7
# 	nn = 3.
# 	NN = 500e3
# 	SS = calc_SS(NN=NN,BB=BB,nn=nn,lbda=CLW,US=Uval,npts=10001)
# 	TT = calc_from_SN(NN,SS,hh=2*CLa,lbda=CLW)
# 	return SS, TT


# def run_ISU_test(Uval):
# 	CLR = .4
# 	CLC = 2.*np.pi*CLR
# 	CLW = .183
# 	CLa = .0153
# 	DC_Tau = 9e3 # [Pa] Wall drag
# 	T_ = model_TS_single(500e3,h=2*CLa,B=6.3e7,nn=3,lbda=CLW,US=Uval,npts=10001)
# 	return T_ + DC_Tau


# def run_ISU_tests():
# 	Uvals = np.linspace(2,200,100)
# 	Tvals = []
# 	for U_ in Uvals:
# 		Tvals.append(run_ISU_test(U_))
# 	return (Uvals,Tvals)



### These are computationally expensive. Just use linear interpolation
# def loop_N2Tau(N_vect,output='tau',**kwargs):
# 	TV = np.zeros(len(N_vect),)
# 	for i_,N_ in enumerate(N_vect):
# 		TV[i_] += model_TS_single(N_,output=output,**kwargs)
# 	return TV

# def loop_N2TauS(N_vect,output='params',**kwargs):
# 	# TV = np.zeros(len(N_vect),)
# 	# SV = np.zeros(len(N_vect),)
# 	for i_, N_ in tqdm(enumerate(N_vect)):
# 		iT,iS = model_TS_single(N_,output=output,**kwargs)
# 		TV[i_] += iT
# 		SV[i_] += iS
# 	return TV, SV




# S = np.linspace(0,1,1000)
