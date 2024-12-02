
"""
:module: src.model.lliboutry_kamb_model
:auth: Nathan T. Stevens
:email: ntsteven@uw.edu (formerly nstevens@wisc.edu)
:org: Pacific Northwest Seismic Network / University of Wisconsin - Madison
:license: Creative Commons Attribution 4.0 (CC-BY-4.0)

:purpose:
	This module hosts methods for calculating  discretized solutions for cavity geometry and resultant
	drag using the double-valued quasi-equilibrium model in Zoet & Iverson (2015) using the analytic solutions
	of Lliboutry (1976) and Kamb (1987)

:references:
Kamb B (1987) Glacier surge mechanism based on linked cavity configuration of the basal water conduit system. 
	Journal of Geophysical Research 92(B9), 9083–9100. doi:10.1029/JB092iB09p09083.	

Lliboutry L (1979) Local Friction Laws For Glaciers: A Critical Review and New Openings. 
	Journal of Glaciology 23(89), 67–95.

Zoet LK and Iverson NR (2015) Experimental determination of a double-valued drag relationship for glacier sliding.
	Journal of Glaciology 61(225), 1–7. doi:10.3189/2015JoG14J174.

ToDos
- TODO: Add full citations for references
"""
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

def calc_N(Pw, HH, rho=917., g=9.81):
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


def model_roof(NN, USS=15./(365*24*3600), lbda=0.31425, amp=0.0253, BB=6.3e7, nn=3., npts=5001):
	"""Model the cavity roof geometry ignoring the presence
	of a down-flow obstacle. Default values set for the 
	UW-CRSD experiments in Stevens and others (in review)

	:param NN: effective pressure [pascals]
	:type NN: float
	:param USS: steady-state sliding velocity,
		defaults to 15./(365*24*3600) [meters per annum]
	:type USS: float, optional
	:param lbda: obstacle wave-length (also model domain length),
		defaults to 0.31425 [meters]
	:type lbda: float, optional
	:param amp: obstacle amplitude, defaults to 0.0253 [meters]
	:type amp: float, optional
	:param BB: ice effective viscosity, defaults to 6.3e7 [Pa annum**1/nn]
	:type BB: float, optional
	:param nn: ice flow exponent, defaults to 3 [dimensionless]
	:type nn: float, optional
	:param npts: number of model points, defaults to 5001 [count]
	:type npts: int, optional
	:returns:
	 - **xvect** (*numpy.ndarray*) -- horizontal coordinate vector [meters]
	 - **gvect** (*numpy.ndarray*) -- vertical coordinate vector [meters]
	"""	
	# Create modeling domain
	xvect = np.linspace(0,lbda,npts)
	# Convert wavelength to wavenumber
	wavenumber = lbda2k(lbda)
	# Calculate cavity lengthscale
	cavity_l = calc_cavity_l(NN=NN, USS=USS, amp=amp, BB=BB, nn=nn)
	# Calculate roof elevation profile
	gvect = calc_g_roof(xvect=xvect, cavity_l=cavity_l, amp=amp)
	return xvect, gvect

def model_bed(lbda=0.31425, amp=0.0253, npts=5001):
	"""Create a discretized model of the UW-CRSD sinusoidal
	bed for one obstacle (crest-to-crest).

	:param lbda: obstacle wavelength, defaults to 0.31425 [meters]
	:type lbda: float, optional
	:param amp: obstacle amplitude, defaults to 0.0253 [meters]
	:type amp: float, optional
	:param npts: number of model points, defaults to 5001 [count]
	:type npts: int, optional
	:returns:
	 - **xvect** (*numpy.ndarray*) -- horizontal coordinate vector [meters]
	 - **hvect** (*numpy.ndarray*) -- vertical coordinate vector [meters]
	"""	
	xvect = np.linspace(0,lbda, npts)
	wavenumber = lbda2k(lbda)
	# Calculate bed elevation profile
	hvect = calc_h_bed(xvect=xvect, amp=amp, wavenumber=wavenumber)
	return xvect, hvect

def get_cavity_ends(xvect, gvect, hvect):
	"""Get the reattachment and detachment point 
	positions of a modeled cavity described by
	a common horizontal sampling vector and two
	model value vectors for the bed surface and
	the cavity roof.

	This assumes ice flow is from left to right
	(i.e., increasing xvect)

	:param xvect: horizontal sampling vector
	:type xvect: numpy.ndarray
	:param gvect: cavity roof elevation vector
	:type gvect: numpy.ndarray
	:param hvect: bed surface elevation vector
	:type hvect: numpy.ndarray
	:returns:
	 - **xr** (*scalar*) -- closest xvect element to the
	 	cavity reattachment point (right edge of cavity)
	 - **xd** (*scalar*) -- closest xvevt element to the
	 	cavity detachment point (left edge of cavity)
	"""	
	if xvect.shape != gvect.shape:
		raise ValueError
	if xvect.shape != hvect.shape:
		raise ValueError
	if gvect.shape != hvect.shape:
		raise ValueError

	idx = np.argwhere(np.diff(np.sign(gvect - hvect))).flatten()
	# If there are two distinct points, parse
	if len(idx) == 2:
		xd = xvect[idx[0]]
		xr = xvect[idx[1]]
	# FIXME: Need to tidy this up
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


def calc_v_equivalent(N, N0, V0, hh=0.0253*2, BB=6.3e7, nn=3):
	"""
	Model the velocity perturbation that would produce a comparable
	steady-state cavity geometry as an effective pressure perturbation
	given a reference effective pressure and velocity

	:param N: new effective pressure [pascals]
	:type N: float
	:param N0: reference effective pressure [pascals]
	:type N0: float
	:param V0: reference sliding velocity [meters annum**-1]
	:type V0: float
	:param hh: bed obstacle height, defaults to 0.0506 [meters]
	:type hh: float, optional
	:param lbda: bed obstacle wavelength, defaults to 0.31425 [meters]
	:type lbda: float, optional
	:param BB: effective viscosity, defaults to 6.3e7 [Pa annum**1/nn]
	:type BB: float, optional
	:param nn: flow law exponent, defaults to 3
	:type nn: float, optional
	:return:
	 - **V** (*float*) -- equivalent velocity
	"""
	# Calculate cavity length
	ll = calc_cavity_l(N, V0, hh/2, BB, nn)
	# Solve for velocity equivalent
	V = (np.pi*ll**2)/(8.*hh*(BB/N0)**nn)
	return V


