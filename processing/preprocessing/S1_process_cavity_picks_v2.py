"""
:module: S1_process_cavity_picks.py
:purpose: Conduct post-processing on manually adjusted and picked locations of
			 ice-bed contact geometries from time-lapse image series by cameras
			 #2 and #4 during experiment T2. See notes on correction approaches
			 below.
:auth: Nathan T. Stevens
:email: ntstevens@wisc.edu
"""


import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
sys.path.append(os.path.join('..','..'))
import util.datetimeindex as dtiu


### SUBROUTINES ####
def reorg_picks(df,camno=None):
	odf = pd.DataFrame()
	for C_ in ['Crest','Stoss','Lee']:
		idf = df.copy()[df['Location']==C_]
		Sx = idf['X']
		Sx.name = '%s X'%(C_)
		Sy = idf['Y']
		Sy.name = '%s Y'%(C_)
		Si = idf['id']
		Si.name = '%s id'%(C_)

		iodf = pd.concat([Si,Sx,Sy],axis=1,ignore_index=False)
		odf = pd.concat([odf,iodf],axis=1,ignore_index=False)
	if camno is not None:
		cam_idx = []
		for i_ in range(len(odf)):
			cam_idx.append(camno)
		odf = pd.concat([odf,pd.Series(cam_idx,odf.index,name='cam')],axis=1,ignore_index=False)
	return odf


### DEFINE DIRECTORY/FILE STRUCTURE ###
# Common root
ROOT = os.path.join('..','..','..')
# Input data directories
IDIR = os.path.join(ROOT,'raw','Cavity_Picks')
DDIR = os.path.join(ROOT,'processed','timeseries')
# Output data directory
ODIR = os.path.join(ROOT,'processed','Cavity_Picks')
T24_LV = os.path.join(DDIR,'S4_T24_LVDT_10min_x3_smoothed_QSS_reduced.csv')
T24_NT = os.path.join(DDIR,'S3_T24_NT_10min_x3_smoothed.csv')
# Manual Pick Data
PICKS = os.path.join(IDIR,'master_picks_T24_Exported_Points_v6.csv')

#### LOAD DATA ####
# Load processed LVDT data
df_Z24 = pd.read_csv(T24_LV,parse_dates=True,index_col=[0])
# Load Effective Pressure and Shear Stress data
df_NT24 = pd.read_csv(T24_NT,parse_dates=True,index_col=[0])


# Do inital load
df_main = pd.read_csv(PICKS,parse_dates=['DateTime'],index_col='DateTime')

# Organize picks into organized sets w/ DateTime indexing
df_main = reorg_picks(df_main)

## PROCESS CAMERA INDICES
# Filter data to specified T24 experiment timestamps bounding Camera 4 imaging
IND_c4 = (df_main.index >= pd.Timestamp("2021-10-28T12")) & (df_main.index <= pd.Timestamp("2021-10-30T13"))
# Create holder to pass labels to data frames
cam = []
for i_ in IND_c4:
	# If timestamp of data point is in IND_c4, label data as cam4
	if i_:
		cam.append('cam4')
	# Otherwise label data as cam2
	else:
		cam.append('cam2')
# Append these labels to the main dataframe
df_main = pd.concat([df_main,pd.Series(cam,index=df_main.index,name='cam')],axis=1,ignore_index=False)




## Get Camera Transfer Indices
for i_ in range(len(IND_c4) - 1):
	if not IND_c4[i_] and IND_c4[i_ + 1]:
		IND_c2_to_c4 = [i_, i_ + 1]
	elif IND_c4[i_] and not IND_c4[i_ + 1]:
		IND_c4_to_c2 = [i_, i_ + 1]


## Calculate and Apply Vertical and Horizontal offsets
df_pcw = pd.DataFrame()
for fld_ in ['Stoss X','Crest X','Lee X','Crest Y','Stoss Y','Lee Y']:
	# Calculate piecewise vertical offsets at camera transitions
	D_c2_to_c4 = df_main[fld_].values[IND_c2_to_c4[1]] - df_main[fld_].values[IND_c2_to_c4[0]]
	D_c4_to_c2 = df_main[fld_].values[IND_c4_to_c2[0]] - df_main[fld_].values[IND_c4_to_c2[1]]
	S_p = pd.concat([df_main[fld_].iloc[:IND_c2_to_c4[1]] + D_c2_to_c4,\
					  df_main[fld_].iloc[IND_c2_to_c4[1]:IND_c4_to_c2[1]],\
					  df_main[fld_].iloc[IND_c4_to_c2[1]:] + D_c4_to_c2],\
					 axis=0,ignore_index=False)
	S_p = pd.Series(S_p,name='%s Piecewise'%(fld_))
	df_pcw = pd.concat([df_pcw,S_p],axis=1,ignore_index=False)

## Calculate Differential Distances
df_dpcw = pd.DataFrame()
for cord_ in ['X','Y']:
	# Apply ad-hoc vertical correction based on observations
	if cord_ == 'Y':
		DC_corr = 0.00
	else:
		DC_corr = 0.
	for prt_ in ['Stoss','Lee']:
		fld_ = prt_ + ' ' + cord_ + ' Piecewise'
		S_dp = df_pcw[fld_] - df_pcw['Crest %s Piecewise'%(cord_)] - DC_corr
		S_dp = pd.Series(S_dp*1e3,name='%s d%s Piecewise mm'%(prt_,cord_))
		df_dpcw = pd.concat([df_dpcw,S_dp],axis=1,ignore_index=False)

## Stitch in camera indices
df_dpcw = pd.concat([df_dpcw,df_main['cam']],axis=1,ignore_index=False)


## Extract nearest LVDT readings to each timestamp
dz = []; N = []; tau = []
for ts_ in df_dpcw.index:
	IND = np.argmin(np.abs((df_Z24.index - ts_).total_seconds()))
	dz.append(df_Z24['LVDT_mm_stitched red'].values[IND])
	IND = np.argmin(np.abs((df_NT24.index - ts_).total_seconds()))
	N.append(df_NT24['N_kPa'].values[IND])
	tau.append(df_NT24['T_kPa'].values[IND])

# Compose a DataFrame for subsequent modeling/visualization
df_out = pd.concat([df_dpcw,pd.DataFrame({'Dz* mm':dz,'N kPa':N,'T kPa':tau},index=df_dpcw.index)],axis=1,ignore_index=False)


### APPLY ROTATION CORRECTIONS ###
DY = -0.25 # [mm] 


df2a = df_out[(df_out['cam']=='cam2')&(df_out.index < pd.Timestamp("2021-10-29"))]
df4 = df_out[df_out['cam']=='cam4']
df2b = df_out[(df_out['cam']=='cam2')&(df_out.index < pd.Timestamp("2021-10-29"))]


def rmat(theta):
	"""
	Compose a 2D rotation matrix taking
	input angles in radians that will apply
	a counterclockwise rotation to [X,Y] data.
	"""
	RR = np.array([[np.cos(theta),-np.sin(theta)],\
				   [np.sin(theta),np.cos(theta)]])
	return RR

def bedmodel_mm(x_mm,h=1,lbda=1):
	"""
	Create a model of the sinusoidal bed
	"""
	y_mm = (h/2*np.cos(2*np.pi*lbda**-1*x_mm) - h/2)
	return y_mm

def res(x,y,h=1,lbda=1):
	y_hat = bedmodel_mm(x,h=h,lbda=lbda)
	res = y - y_hat
	return res

# SUBSET SCENES
IND2a = (df_out['cam']=='cam2')&(df_out.index < pd.Timestamp("2021-10-29"))
IND2b = (df_out['cam']=='cam2')&(df_out.index > pd.Timestamp("2021-10-29"))
IND4 = df_out['cam']=='cam4'

df_X2aL = pd.DataFrame({'X':df_out[IND2a]['Lee dX Piecewise mm'].values,\
					 	'Y':df_out[IND2a]['Lee dY Piecewise mm'].values})
df_X2bL = pd.DataFrame({'X':df_out[IND2b]['Lee dX Piecewise mm'].values,\
					 	'Y':df_out[IND2b]['Lee dY Piecewise mm'].values})
df_X4L = pd.DataFrame({'X':df_out[IND4]['Lee dX Piecewise mm'].values,\
					   'Y':df_out[IND4]['Lee dY Piecewise mm'].values})
df_X2aS = pd.DataFrame({'X':df_out[IND2a]['Stoss dX Piecewise mm'].values,\
					 'Y':df_out[IND2a]['Stoss dY Piecewise mm'].values})
df_X2bS = pd.DataFrame({'X':df_out[IND2b]['Stoss dX Piecewise mm'].values,\
					 'Y':df_out[IND2b]['Stoss dY Piecewise mm'].values})
df_X4S = pd.DataFrame({'X':df_out[IND4]['Stoss dX Piecewise mm'].values,\
					 'Y':df_out[IND4]['Stoss dY Piecewise mm'].values})


## Find optimal rotation to minimize data-model misfits

BED = {'h': 0.078e3, 'lbda': 0.47123889803846897e3}
bedX = np.linspace(-0.5*BED['lbda'],0.5*BED['lbda'],201)
bedY = bedmodel_mm(bedX,**BED)

Data_Tuple = (df_X2aL,df_X2aS,df_X4L,df_X4S,df_X2bL,df_X2bS)


Ovect = np.linspace(-6,6,501)*(np.pi/180.)
DYvect = np.linspace(-3,3,501)

print('Running rotation fitting')
### DO JUST ROTATION OPTIMIZATION PER DATA SET ###
holder = []
for O_ in tqdm(Ovect):
	line = []
	for D_ in Data_Tuple:
		xr_ = np.matmul(rmat(O_),D_.values.T)
		rL2 = np.linalg.norm(res(xr_[0,:],xr_[1,:],**BED))
		line.append(rL2)
	holder.append(line)


df_res = pd.DataFrame(holder,columns=['cam2a lee','cam2a stoss','cam4 lee','cam4 stoss','cam2b lee','cam2b stoss'],\
					  index=Ovect)

Eopt = {}
Edat = {}
plt.figure()
plt.subplot(121)
for C_ in df_res.columns:
	plt.plot(df_res.index*(180./np.pi),df_res[C_],label=C_)
	IND_ = df_res[C_] == df_res[C_].min()
	Eopt.update({C_:df_res.index[IND_].values[0]})
	plt.plot(df_res[IND_].index*(180./np.pi),df_res[IND_][C_],'ro')
plt.legend()
plt.xlabel('Rotation angle [$\\degree$]')
plt.ylabel('L-2 Norm Residual [mm]')

plt.subplot(122)
plt.plot(bedX,bedY,'k',label='Outer wall bed profile')
leeX,leeY,stossX,stossY = [],[],[],[]
for i_,C_ in enumerate(df_res.columns):
	xr_ = np.matmul(rmat(Eopt[C_]),Data_Tuple[i_].values.T)
	x_ = xr_[0,:]
	y_ = xr_[1,:]
	plt.plot(x_,y_,'.',label=C_)
	if 'lee' in C_:
		leeX += list(x_)
		leeY += list(y_)
	if 'stoss' in C_:
		stossX += list(x_)
		stossY += list(y_)

	Edat.update({C_+' dX mm':x_,C_+' dY mm':y_})
plt.legend()
plt.xlabel('X-offset from bump crest [mm]')
plt.ylabel('Y-offset from bump crest [mm]')



df_rot = pd.DataFrame({'Lee dX PR mm':leeX,'Lee dY PR mm':leeY,'Lee dY PRF mm':bedmodel_mm(np.array(leeX),**BED),\
					   'Stoss dX PR mm':stossX,'Stoss dY PR mm':stossY,'Stoss dY PRF mm':bedmodel_mm(np.array(stossX),**BED)},\
					  index=df_main.index)


### DO ROTATION AND SUBSEQUENT VERTICAL SHIFT GRIDSEARCH ###

def argmin_nD(x):
	out = (x_[0] for x_ in np.where(x == np.min(x)))
	return out


print('Doing rotation and vertical offset grid search for lee data')
Ovect = np.linspace(-6,6,501)*(np.pi/180.)
DYvect = np.linspace(-3,3,401)

## Conduct 2-parameter grid-search for lee contacts
holder = []
for O_ in tqdm(Ovect):
	for Y_ in DYvect:
		line = [O_,Y_]
		for D_ in Data_Tuple[::2]:
			xr_ = np.matmul(rmat(O_),D_.values.T)
			rL2 = np.linalg.norm(res(xr_[0,:],xr_[1,:]+Y_,**BED))
			line.append(rL2)
		holder.append(line)


df_resOY = pd.DataFrame(holder,columns=['thetaL','offsetL',\
										'cam2a','cam4','cam2b'])

print('Doing follow-up rotation search for stoss contacts \n with best-fit vertical fit from same contacts')
holder = []
df_lee_bf = pd.DataFrame()
i_ = 0
for O_ in tqdm(Ovect):
	line = [O_]
	for D_,E_ in [('cam2a',Data_Tuple[1]),('cam4',Data_Tuple[3]),('cam2b',Data_Tuple[5])]:
		if i_ == 0:
			# Find best-fit value from 2-parameter sesarch
			imins = df_resOY[df_resOY[D_] == df_resOY[D_].min()][['thetaL','offsetL']]
			df_imin = imins.copy()
			df_imin.index = [D_]
			df_lee_bf = pd.concat([df_lee_bf,df_imin],axis=0,ignore_index=False)
		
		ix_ = E_['X'].values
		iy_ = E_['Y'].values + imins['offsetL'].values[0]
		ixy_ = np.c_[ix_,iy_]
		xyr_ = np.matmul(rmat(O_),ixy_.T)
		rL2 = np.linalg.norm(res(xyr_[0,:],xyr_[1,:],**BED))
		line.append(rL2)
	i_ += 1
	holder.append(line)

# Compile data into dataframe
df_resO = pd.DataFrame(holder,\
					   columns=['thetaS','cam2a','cam4','cam2b']).\
			 set_index('thetaS')

df_sto_bf = pd.DataFrame()
for C_ in df_resO.columns:
	S_min = pd.Series(df_resO[df_resO[C_] == df_resO[C_].min()].index)
	S_min.index = [C_]
	df_sto_bf = pd.concat([df_sto_bf,S_min],axis=0)

df_sto_bf = df_sto_bf.rename(columns={0:'thetaS'})

df_bf = pd.concat([df_lee_bf,df_sto_bf],axis=1,ignore_index=False)






### PLOTTING SECTION ###
def argmin_nD(x):
	out = [x_[0] for x_ in np.where(x == np.min(x))]
	return out

OO,YY = np.meshgrid(Ovect,DYvect)

fig,axs = plt.subplots(nrows=3,ncols=3)


ZZ2aL = np.reshape(df_resOY['cam2a'].values,(len(Ovect),len(DYvect))).T
axs[0,0].pcolor(OO,YY,ZZ2aL)
cs = axs[0,0].contour(OO,YY,ZZ2aL,colors='k')
plt.clabel(cs)
Imin2aL = argmin_nD(ZZ2aL)
axs[0,0].plot(OO[Imin2aL[0],Imin2aL[1]],YY[Imin2aL[0],Imin2aL[1]],'r*')

axs[1,0].plot(df_resO['cam2a'])
axs[1,0].plot(df_sto_bf.T['cam2a'],df_resO['cam2a'].min(),'r*')


# ZZ2aS = np.reshape(df_resOY['cam2a stoss'].values,(len(Ovect),len(DYvect))).T
# axs[0,1].contourf(OO,YY,ZZ2aS)
# Imin2aS = argmin_nD(ZZ2aS)
# axs[0,1].plot(OO[Imin2aS[0],Imin2aS[1]],YY[Imin2aS[0],Imin2aS[1]],'r*')


ZZ4L = np.reshape(df_resOY['cam4'].values,(len(Ovect),len(DYvect))).T
axs[0,1].pcolor(OO,YY,ZZ4L)
cs = axs[0,1].contour(OO,YY,ZZ4L,colors='k')
plt.clabel(cs)
Imin4L = argmin_nD(ZZ4L)
axs[0,1].plot(OO[Imin4L[0],Imin4L[1]],YY[Imin4L[0],Imin4L[1]],'r*')

axs[1,1].plot(df_resO['cam4'])
axs[1,1].plot(df_sto_bf.T['cam4'],df_resO['cam4'].min(),'r*')


# ZZ4S = np.reshape(df_resOY['cam4 stoss'].values,(len(Ovect),len(DYvect))).T
# axs[1,1].contourf(OO,YY,ZZ4S)
# Imin4S = argmin_nD(ZZ4S)
# axs[1,1].plot(OO[Imin4S[0],Imin4S[1]],YY[Imin4S[0],Imin4S[1]],'r*')


ZZ2bL = np.reshape(df_resOY['cam2b'].values,(len(Ovect),len(DYvect))).T
axs[0,2].pcolor(OO,YY,ZZ2bL)
cs = axs[0,2].contour(OO,YY,ZZ2bL,colors='k')
Imin2bL = argmin_nD(ZZ2bL)
axs[0,2].plot(OO[Imin2bL[0],Imin2bL[1]],YY[Imin2bL[0],Imin2bL[1]],'r*')

axs[1,2].plot(df_resO['cam2b'])
axs[1,2].plot(df_sto_bf.T['cam2b'],df_resO['cam2b'].min(),'r*')

# ZZ2bS = np.reshape(df_resOY['cam2b stoss'].values,(len(Ovect),len(DYvect))).T
# axs[2,1].contourf(OO,YY,ZZ2bS)
# Imin2bS = argmin_nD(ZZ2bS)
# axs[2,1].plot(OO[Imin2bS[0],Imin2bS[1]],YY[Imin2bS[0],Imin2bS[1]],'r*')

# Label Cost Function Plots
for i_ in range(3):
	for j_ in range(3):
		if i_ == 0:
			axs[i_,j_].set_title(df_bf.index[j_]) 
			if j_ == 0:
				axs[i_,j_].set_ylabel('Vertical offset [mm]')
		if i_ == 1:
			axs[i_,j_].set_xlabel('Rotation angle [rad]')
		if i_ == 2:
			fig.delaxes(axs[i_,j_])

## Re-constitute best-fit model and plot
ax7 = fig.add_subplot(313)

ax7.plot(bedX,bedY,'k-',zorder=2)

XLc,YLc,XSc,YSc = [],[],[],[]

for i_,I_ in enumerate(df_bf.index):
	S_bf = df_bf.T[I_]
	idf_lee = Data_Tuple[2*i_]
	idf_sto = Data_Tuple[2*i_ + 1]
	# Process best-fit lee correction model for scene I_
	xyl_ = idf_lee.values.T
	# First rotate lee
	xylr_ = np.matmul(rmat(S_bf['thetaL']),xyl_)
	# Then apply offset
	xylr_[1,:] += S_bf['offsetL']
	# Append results to lists
	XLc += list(xylr_[0,:])
	YLc += list(xylr_[1,:])
	# Process best-fit stoss correction model for scene I_
	xys_ = idf_sto.values.T
	# First apply offset
	xys_[1,:] += S_bf['offsetL']
	# Then apply rotation
	xysr_ = np.matmul(rmat(S_bf['thetaS']),xys_)
	# Append results to lists
	XSc += list(xysr_[0,:])
	YSc += list(xysr_[1,:])

	ax7.plot(list(xyl_[0,:]) + list(xys_[0,:]),\
			 list(xyl_[1,:]) + list(xys_[1,:]),\
			 's',label=I_)

	ax7.plot(list(xylr_[0,:]) + list(xysr_[0,:]),\
			 list(xylr_[1,:]) + list(xysr_[1,:]),\
			 '.',label=I_)




df_corrected = pd.DataFrame({'Stoss dX POR mm':XSc,'Stoss dY POR mm':YSc,\
							 'Lee dX PRO mm':XLc,'Lee dY PRO mm':YLc},\
							index=df_out.index)

df_out = pd.concat([df_out,df_rot,df_corrected],axis=1,ignore_index=False)
plt.show()
# breakpoint()


# Save data
df_out.to_csv(os.path.join(ODIR,'Postprocessed_Contact_Geometries.csv'),header=True,index=True)




# for i_ in range(2):
# 	for j_ in range(3):
# 		axs[i_,j_].set_xlabel('Rotation angle [rad]')

# 		if i_ == 0:
# 			axs[i_,j_].set_ylabel('Vertical offset [mm]')
# 			if j_ == 0:
# 				axs[i_,j_].set_title('Lee Optimization')
# 		fig.delaxes(axs[2,j_])
# 		if i_ == 1:	
# 			axs[i_,j_].set_ylabel('|res|$_2$ [mm]')
# 			if j_ == 0:
# 				axs[i_,j_].set_title('Stoss Optimization')
# 			axs[i_,j_].yaxis.set_ticks_position('right')
# 			axs[i_,j_].yaxis.set_label_position('right')


			



# df_out = pd.concat([df_out,df_rot],axis=1,ignore_index=False)





# plt.figure()
# # Plot outer wall bed profile
# plt.plot(bedX,bedY,'k')
# # Plot X2a
# plt.plot(df_X2a['X'],df_X2a['Y'],'.',label='cam2a')
# # Plot X2b
# plt.plot(df_X2b['X'],df_X2b['Y'],'.',label='cam2b')
# # Plot X4
# plt.plot(df_X4['X'],df_X4['Y'],'.',label='cam4')



























# # If isplot, do some modeling/statistical tests
# isplot = True
# # NTS: This will be obsolited here and passed to another processing
# # script that tests a number of hypotheses
# if isplot:
# 	### Conduct some statistical tests on model fits 
# 	y = df_out[['Lee dY Piecewise mm','Stoss dY Piecewise mm',\
# 				'Lee dX Piecewise mm','Stoss dX Piecewise mm']].values

# 	mod1,cov1 = np.polyfit(dz,y,1,cov=True)
# 	mod2,cov2 = np.polyfit(dz,y,2,cov=True)
# 	mod3,cov3 = np.polyfit(dz,y,3,cov=True)
# 	mod4,cov4 = np.polyfit(dz,y,4,cov=True)

# 	res1, res2, res3, res4 = [], [], [], []
# 	yhat1, yhat2, yhat3, yhat4 = [], [], [], []

# 	for i_ in range(4):
# 		imod1 = mod1[:,i_]
# 		imod2 = mod2[:,i_]
# 		imod3 = mod3[:,i_]
# 		imod4 = mod4[:,i_]
# 		yhat1.append(np.poly1d(imod1)(dz))
# 		yhat2.append(np.poly1d(imod2)(dz))
# 		yhat3.append(np.poly1d(imod3)(dz))
# 		yhat4.append(np.poly1d(imod4)(dz))	
# 		res1.append(y[:,i_] - np.poly1d(imod1)(dz))
# 		res2.append(y[:,i_] - np.poly1d(imod2)(dz))
# 		res3.append(y[:,i_] - np.poly1d(imod3)(dz))
# 		res4.append(y[:,i_] - np.poly1d(imod4)(dz))


# 	plt.figure()
# 	for i_,S_ in enumerate(['dY lee','dY stoss','dX lee','dX stoss']):
# 		plt.plot(df_out.index,yhat1[i_],'.',label=S_)
# 		plt.plot(df_out.index,y[:,i_],'s',label=S_)

# 	plt.legend()

# 	plt.show()

# plt.figure()
# for i_,fld_ in enumerate(['Lee dY Piecewise mm','Stoss dY Piecewise mm','Lee dX Piecewise mm','Stoss dX Piecewise mm']):
# 	x = df_out[fld_].values
# 	plt.subplot(221+i_)
# 	plt.plot(dz,x,'o',label=fld_)
# 	for deg in [1,2,3]:
# 		# Calculate polynomail model fit
# 		mod,cov = np.polyfit(dz,x,deg,cov=True)
# 		res = dz - np.poly1d(mod)(dz)
# 		cc = np.corrcoef(dz,x)[0,1]
# 		# Calculate the Coefficient of Determination (R**2)
# 		R2 = 1. - np.sum(res**2)/np.sum(res)**2
# 		# Plot modeled vectors
# 		xmin = np.min(dz); xmax = np.max(dz)
# 		plt.plot(np.linspace(xmin,xmax,100),np.poly1d(mod)(np.linspace(xmin,xmax,100)),\
# 				 label='%d: $R^2$ %.4e\nCC: %.2e'%(deg,R2,cc))
# 	plt.legend()

# plt.figure()

# plt.subplot(133)
# x = df_out['Stoss dY Piecewise mm'].values - df_out['Lee dY Piecewise mm'].values
# plt.plot(dz,x,'o',label='Total Contact Area')
# for deg in [1,2,3]:
# 	mod,cov = np.polyfit(dz,x,deg,cov=True)
# 	res = dz - np.poly1d(mod)(dz)
# 	cc = np.corrcoef(dz,x)[0,1]
# 	# Calculate the Coefficient of Determination (R**2)
# 	R2 = 1. - np.sum(res**2)/np.sum(res)**2
# 	xmin = np.min(dz); xmax = np.max(dz)
# 	plt.plot(np.linspace(xmin,xmax,100),np.poly1d(mod)(np.linspace(xmin,xmax,100)),\
# 			 label='%d: $R^2$ %.4e\nCC: %.2e'%(deg,R2,cc))

# plt.legend()
# ### Do linear model of \Delta z^{*}(t)

# MOD,COV = np.polyfit(dz,df_out['Stoss dY Piecewise mm'].values,1,cov=True)

# plt.figure()
# plt.subplot(211)
# plt.plot(df_out['Stoss dY Piecewise mm'],'.-')
# plt.plot(df_out.index,np.poly1d(MOD)(dz),'s')

## TODO: Do Orthogonal Distance Regression Fitting for a linear model




# plt.figure()
# df_dpcw.plot()
# plt.show()



# ## Calculate raw stoss & lee lengths for each camera
# df_stoss = pd.concat([df_main[['Stoss X','Stoss Y','Crest X','Crest Y','cam']],\
# 							  pd.Series(df_main['Crest X'] - df_main['Stoss X'],name='Stoss dx'),\
# 							  pd.Series(df_main['Stoss Y'] - df_main['Crest Y'],name='Stoss dy')],\
# 							 axis=1,ignore_index=False)
# df_lee = pd.concat([df_main[['Lee X','Lee Y','Crest X','Crest Y','cam']],\
# 							  pd.Series(df_main['Lee X'] - df_main['Crest X'],name='Lee dx'),\
# 							  pd.Series(df_main['Lee Y'] - df_main['Crest Y'],name='Lee dy')],\
# 							 axis=1,ignore_index=False)



# ## Extract distance data
# # Get Stoss delta X values
# S_dxs = df_stoss['Stoss dx']
# S_dys = df_stoss['Stoss dy']
# L_dxs = df_lee['Lee dx']
# L_dys = df_lee['Lee dy']


# # Extract reference locations for X coordinates
# # reference from 
# sxref0 = S_dxs.values[loc1[0]]
# sxref1 = S_dxs.values[loc1[1]]
# sxref2 = S_dxs.values[loc2[0]]
# sxref3 = S_dxs.values[loc2[1]]

# syref0 = S_dys.values[loc1[0]]
# syref1 = S_dys.values[loc1[1]]
# syref2 = S_dys.values[loc2[0]]
# syref3 = S_dys.values[loc2[1]]

# S2_dxs = S_dxs[~IND_c4]
# S2_dys = S_dys[~IND_c4]
# S4_dxs = S_dxs[IND_c4]
# S4_dys = S_dys[IND_c4]


# lxref0 = L_dxs.values[loc1[0]]
# lxref1 = L_dxs.values[loc1[1]]
# lxref2 = L_dxs.values[loc2[0]]
# lxref3 = L_dxs.values[loc2[1]]

# lyref0 = L_dys.values[loc1[0]]
# lyref1 = L_dys.values[loc1[1]]
# lyref2 = L_dys.values[loc2[0]]
# lyref3 = L_dys.values[loc2[1]]

# L2_dxs = L_dxs[~IND_c4]
# L2_dys = L_dys[~IND_c4]
# L4_dxs = L_dxs[IND_c4]
# L4_dys = L_dys[IND_c4]


# ### STOSS PROCESSING ###
# # Calculate camera 2 horizontal corrections 
# DXC1 = sxref0 - sxref1
# DXC2 = sxref3 - sxref2
# # Calculate camera 2 vertical corrections
# DYC1 = syref0 - syref1
# DYC2 = syref3 - syref2

# ## Create offset-corrected Camera 2 Stoss elements
# # X-components
# S_c2s1_dxs = S_dxs[S_dxs.index <= S_dxs.index[loc1[0]]] - DXC1
# S_c2s2_dxs = S_dxs[S_dxs.index >= S_dxs.index[loc2[1]]] - DXC2
# # Y-components
# S_c2s1_dys = S_dys[S_dys.index <= S_dys.index[loc1[0]]] - DYC1
# S_c2s2_dys = S_dys[S_dys.index >= S_dys.index[loc2[1]]] - DYC2

# ## Put together series with Camera 4 observations
# Sx_DC = pd.concat([S_c2s1_dxs,S4_dxs,S_c2s2_dxs],axis=0).sort_index()
# Sx_DC.name = 'Stoss X Piecewise'

# Sy_DC = pd.concat([S_c2s1_dys,S4_dys,S_c2s2_dys],axis=0).sort_index()
# Sy_DC.name = 'Stoss Y Piecewise'
