"""
plot_ringshear_figure.py

This script produces a SIMPLE 3-D rendering of the UW-Madison cryogenic ring-shear device sample chamber

:AUTH: Nathan T. Stevens
:EMAIL: ntstevens@wisc.edu
:VERSION: 0.0
:LAST EDIT: August 6. 2021
"""
import argparse, logging, os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Logger = logging.getLogger('Figure 1C Render')



def thetaR2XY(theta,r):
	x,y = r*np.cos(theta), r*np.sin(theta)
	return x,y

def slopefun(ri,ro,ai,ao,r):
	m = (ao - ai)/(ro - ri)
	b = ai - m*ri
	return m*r + b

def bedline(r,ri,ro,ai,ao,w,theta):
	A = slopefun(ri,ro,ai,ao,r)
	Z = A*(np.cos(w*theta)) + ao
	return Z

def bedsurf(ri,ro,ai,ao,w,r,theta):
	A = slopefun(ri,ro,ai,ao,r)
	Z = A*(np.cos(w*theta)) + ao
	return Z

def generate_sinusoidal_bed(w,ri,ro,ai,ao,dr,dT):
	R,T = np.meshgrid(np.arange(ri,ro + dr,dr),np.arange(0,2*np.pi + dT,dT))
	A = bedsurf(ri,ro,ai,ao,w,R,T)
	return A,R,T

def generate_sinusoidal_trace(w,r,ri,ro,ai,ao,dT):
	R,T = np.meshgrid(r,np.arange(0,2*np.pi + dT, dT))
	A = bedsurf(ri,ro,ai,ao,w,R,T)
	return A,R,T

def generate_cylinder_surface(r,h0,h1,dh,dT):
	H,T = np.meshgrid(np.arange(h0,h1+dh,dh),np.arange(0,2*np.pi + dT,dT))
	R = np.ones(T.shape)*r
	return R,T,H

def generate_circle_surface(z,ri,ro,dr,dT):
	R,T = np.meshgrid(np.arange(ri,ro + dr,dr),np.arange(0,2*np.pi + dT,dT))
	Z = np.ones(T.shape)*z
	return R,T,Z

def generate_radial_slice(theta,r0,r1,dr,h0,h1,dh):
	R,H = np.meshgrid(np.arange(r0,r1+dr,dr),np.arange(h0,h1+dh,dh))
	T = np.ones(R.shape)*theta
	return R,H,T

def generate_ring(r,dT,h0,dh):
	T,H = np.meshgrid(np.arange(0,2*np.pi + dT,dT),np.arange(h0-dh,h0+2*dh,dh))
	R = np.ones(T.shape)*r
	return R,T,H


def main(args):
	w = 4. 		# Wavenumber of installed bed
	ri = 0.1	# [m] Inner Radius of Experimental Chamber
	ro = 0.3 	# [m] Outer Radius of Experimental Chamber
	rc = (ri + ro)/2 # [m] centerline radius
	dr = (ro-ri)/200 # [m] radial element lengths
	dT = (2*np.pi)/200 # [rad] angular element lengths
	ai = .025/2	# [m] Inner Amplitude of Sinusoidal Bed
	ao = .076/2	# [m] Outer Amplitude of Sinusoidal Bed
	h0 = 0.0  	# [m] Bottom elevation of experimetal chamber
	h1 = 0.27   # [m] Top elevation of experimental chamber
	dh = (h1 - h0)/200 # [m] hight element lengths

	fig = plt.figure(figsize=(6,4))
	ax = fig.add_subplot(projection='3d')


	# Generate Bed Geometry
	A,R,T = generate_sinusoidal_bed(w,ri,ro,ai,ao,dr,dT)
	X,Y = thetaR2XY(T,R)
	ax.plot_surface(X,Y,A,color='gray',alpha=0.8)
	# Generate Centerline Trace
	A,R,T = generate_sinusoidal_trace(w,rc,ri,ro,ai,ao,dT)
	X,Y = thetaR2XY(T,R)
	# IDX = (R <= rc*1.0001) & (R >= 0.9999)
	# ax.plot_surface(X[IDX],Y[IDX],A[IDX],color='gray',alpha=0.8)
	ax.plot_surface(X,Y,A+0.01,color='cyan',alpha=0.8)

	# Generate Cylinder Lower, Outer
	R,T,H = generate_cylinder_surface(ro,h0,h1,ao/20,dT)
	X,Y = thetaR2XY(T,R)
	ax.plot_surface(X,Y,H,color='dodgerblue',alpha=0.25)
	# Generate Cylinder Lower, Inner
	R,T,H = generate_cylinder_surface(ri,h0,h1,ao/20,dT)
	X,Y = thetaR2XY(T,R)
	ax.plot_surface(X,Y,H,color='dodgerblue',alpha=0.5)
	# Generate Floor
	R,T,Z = generate_circle_surface(0,ri,ro,dr,dT)
	X,Y = thetaR2XY(T,R)
	ax.plot_surface(X,Y,Z,color='darkgrey',alpha=0.75)
	# Generate Roof
	R,T,Z = generate_circle_surface(h1,ri,ro,dr,dT)
	X,Y = thetaR2XY(T,R)
	ax.plot_surface(X,Y,Z,color='dodgerblue',alpha=0.25)

	# Generate outer reference ring
	R,T,H = generate_ring(ro,dT,ao,0.001)
	X,Y = thetaR2XY(T,R)
	ax.plot_surface(X,Y,H,color='k',alpha=1)
	# Generate centerline reference ring
	R,T,H = generate_ring(ri,dT,ao,0.001)
	X,Y = thetaR2XY(T,R)
	ax.plot_surface(X,Y,H,color='k',alpha=1)
	# Generate inner reference ring
	R,T,H = generate_ring(ri,dT,ao,0.001)
	X,Y = thetaR2XY(T,R)
	ax.plot_surface(X,Y,H,color='k',alpha=1)


	# # Plot example slice
	# theta = np.pi*(1.5 + 1/(2*w))
	# R,H,T = generate_radial_slice(theta,ri - .1*ro,ro*1.1,dr,h0-(h1*.25),h1*1.25,dh)
	# X,Y = thetaR2XY(T,R)
	# idx = [0,-1,0,-1],[0,0,-1,-1]
	# ax.plot_surface(X[idx].reshape(2,2),Y[idx].reshape(2,2),H[idx].reshape(2,2),color='red',alpha=0.25)

	# Set Scale
	# ax.set_xlim([-(ro+ri),(ro+ri)])
	# ax.set_ylim([-(ro+ri),(ro+ri)])
	# ax.set_zlim([-(ro+ri)+(h1 - h0)/2,(ro+ri)+(h1 - h0)/2])
	ax.set_xlim([-ro*0.8, ro*0.8])
	ax.set_ylim([-ro*0.8, ro*0.8])
	ax.set_zlim([-(ro*0.8) + dh*100, ro*0.8 + dh*100])
	ax.set_xlabel('X [m]')
	ax.set_ylabel('Y [m]')
	ax.set_zlabel('Z [m]')
	ax.view_init(azim=-29, elev=16)




	if args.dpi == 'figure':
		dpi = 'figure'
	else:
		try:
			dpi = int(args.dpi)

		except:
			dpi = 'figure'
	if dpi == 'figure':
		savename = os.path.join(args.output_path, f'JGLAC_Fig01C_fdpi.{args.format}')
	else:
		savename = os.path.join(args.output_path, f'JGLAC_Fig01C_{dpi}dpi.{args.format}')
	if not os.path.exists(os.path.split(savename)[0]):
		os.makedirs(os.path.split(savename)[0])
	plt.savefig(savename, dpi=dpi, format=args.format)

	if args.show:
		plt.show()

	# ### PLOT CROSS-SECTION
	# plt.figure()
	# plt.fill_between([ri,ro],[ao-ai,0],[h1,h1],alpha=0.5,color='dodgerblue')
	# plt.fill_between([ri,ro],[0,0],[ao-ai,0],alpha=0.5,color='darkgrey')
	# plt.fill_between([ri,ro],[0,0],[ai + ao,2*ao],alpha=0.5,color='darkgrey')
	# plt.plot([ri,ro],[ao,ao],'k-')
	# plt.plot(np.ones(2)*(ro+ri)/2,[ao,ao+(ao+ai)/2],'k-')
	# plt.plot([ri,ri,ro,ro],[h1,0,0,h1],'k-',linewidth=2)
	# plt.xlim([0,ro*1.1])
	# plt.xlabel('Radius [m]')
	# plt.ylabel('Z [m]')


if __name__ == '__main__':

	parser = argparse.ArgumentParser(
		prog='JGLAC_Fig01C.py',
		description='A simple 3D rendering of the experimental chamber and undulatory bed'
	)
	parser.add_argument(
		'-o',
		'--output_path',
		action='store',
		dest='output_path',
		default='../results/figures',
		help='path and name to save the rendered figure to, minus format (use -f for format). Defaults to "../results/figures/JGLAC_Fig01c"',
		type=str
	)

	parser.add_argument(
		'-f',
		'-format',
		action='store',
		dest='format',
		default='png',
		choices=['png','pdf','svg'],
		help='the figure output format (e.g., *.png, *.pdf, *.svg) callable by :meth:`~matplotlib.pyplot.savefig`. Defaults to "png"',
		type=str
	)

	parser.add_argument(
		'-d',
		'--dpi',
		action='store',
		dest='dpi',
		default='figure',
		help='set the `dpi` argument for :meth:`~matplotlib.pyplot.savefig. Defaults to "figure". All numeric values parsed as int',
	)

	parser.add_argument(
		'-s',
		'--show',
		action='store_true',
		dest='show',
		help='if included, render the figure on the desktop in addition to saving to disk'
	)

	args = parser.parse_args()
	main(args)
