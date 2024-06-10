"""
:module: JGLAC_Fig02_Steady_State_Parameter_Space.py
:purpose: Plot the modeled parameter space (N,\\tau,S,\\mu) for the UW-CRSD
		  assuming rheologic properties identical to those in Zoet & Iverson
		  (2015)
:version: 0 - Submission format to Journal of Glaciology
:short ref: Stevens, Hansen, and others
			Experimental constraints on transient glacier sliding with ice-bed separation
:Figure #: 2
:Figure Caption: Parameter space for the double-valued drag sliding model of Zoet and Iverson (2015) 
				 for the geometry of the UW¬–CRSD and the sinusoidal bed in this study (Table 1) and 
				 UW–CRSD operational limits (Table 2). 
				 	Figure axes show linear slip velocities, V, and effective pressures, N. 
				 	Shading shows predicted \\mu (colorbar) 
				 	solid contours show predicted shear stresses, \tau, and 
				 	dotted contours represent predicted the ice-bed contact area fractions, S. 
				 	The operational limit \tau_{max} = 275 kPa is shown as a red dashed line. 
				 	The operational range of N(t) for these experiments are shown as an orange
				 	line on the centerline velocity, V = 15 m a^{-1} and surrounded by an orange
				 	shaded region that bounds the inner and outer diameter velocities V\\in[X, Y]
:auth: Nathan T. Stevens
:email: ntstevens@wisc.edu

"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
sys.path.append(os.path.join('..'))
import scripts.model.steadystate_model as ssm

issave = True
DPI = 200
FMT = 'PNG'
hires = True

ROOT = os.path.join('..','..')
# NDAT = os.path.join(ROOT,'processed','model','model_results_N_Pa.csv')
SDAT = os.path.join(ROOT,'processed','model','model_results_S.csv')
TDAT = os.path.join(ROOT,'processed','model','model_results_T_Pa.csv')
ODIR = os.path.join(ROOT,'results','figures','manuscript')

# Load Modeled Values
# df_N = pd.read_csv(NDAT)
df_S = pd.read_csv(SDAT,index_col=[0])
df_T = pd.read_csv(TDAT,index_col=[0])

# Create Plotting Meshes
Vv = np.array(df_S.index)
Nv = np.array(df_S.columns.astype(float))*1e-3

Vm,Nm = np.meshgrid(Vv,Nv)


# Calculate Drag
mu_m = []
for c_ in tqdm(df_T.columns):
	mu_i = df_T[c_].values / float(c_)
	mu_m.append(mu_i)
# Format matrix as array
mu_m = np.array(mu_m)
# Turn 0-valued mu estimates into nan values
# ...This corresponds to 
mu_m[mu_m < 1e-9] = np.nan

# Define subsampling interval to speed up plotting
if hires:
	dsx = dsy = 1
else:
	dsx = dsy = 5

# Initialize Figure
plt.figure(figsize=(6,4.5))



# Plot Drag
plt.pcolor(Vm[::dsx,::dsy],Nm[::dsx,::dsy],mu_m[::dsx,::dsy],cmap='Blues',zorder=1)
plt.colorbar()
plt.text(38,500,'Drag ($\\mu$) [ - ]',rotation=270,fontsize=10,ha='center',va='center')
plt.clim([0,.35])
cxloc = 9



# Plot S contours
chdl = plt.contour(Vm[::dsx,::dsy],Nm[::dsx,::dsy],df_S.values.T[::dsx,::dsy],\
				   levels=np.linspace(0,0.8,9),colors='w',linestyles=':',zorder=3)
mlocs = []; 
for level in chdl.collections:
	path = level.get_paths()
	if len(path) > 0:
		cxpath = path[0].vertices[:,0]
		cypath = path[0].vertices[:,1]
		cxidx = np.argmin(np.abs(cxpath - cxloc))
		cyloc = cypath[cxidx]
		mlocs.append((cxloc,cyloc))
		# mlocs.append((np.mean(path[0].vertices[:,0]),\
		# 	 		  np.mean(path[0].vertices[:,1])))

plt.clabel(chdl,inline=True,inline_spacing=2,fontsize=10,fmt='%.1f',manual=mlocs)


# Plot \\tau contours
cxloc = 23
chdl = plt.contour(Vm[::dsx,::dsy],Nm[::dsx,::dsy],df_T.values.T[::dsx,::dsy]*1e-3,\
					levels=np.arange(25,275,25),colors='k',zorder=2)
# Set manual locations (Solution from ChatGPT prompt)
mlocs = []; 
for level in chdl.collections:
	path = level.get_paths()
	if len(path) > 0:
		cxpath = path[0].vertices[:,0]
		cypath = path[0].vertices[:,1]
		cxidx = np.argmin(np.abs(cxpath - cxloc))
		cyloc = cypath[cxidx]
		mlocs.append((cxloc,cyloc))
		# mlocs.append((np.mean(level.get_paths()[0].vertices[:,0]),\
		# 	 		  np.mean(level.get_paths()[0].vertices[:,1])))
plt.clabel(chdl,inline=True,inline_spacing=2,fontsize=10,fmt='%d kPa',manual=mlocs)



cxloc = 23
# Plot \\tau_{max}
chdl = plt.contour(Vm[::dsx,::dsy],Nm[::dsx,::dsy],df_T.values.T[::dsx,::dsy]*1e-3,\
					levels=[275],colors='r',linewidths=2,linestyles='--',zorder=4)
mlocs = []; 
for level in chdl.collections:
	path = level.get_paths()
	if len(path) > 0:
		cxpath = path[0].vertices[:,0]
		cypath = path[0].vertices[:,1]
		cxidx = np.argmin(np.abs(cxpath - cxloc))
		cyloc = cypath[cxidx]
		mlocs.append((cxloc,cyloc))
plt.clabel(chdl,inline=True,inline_spacing=2,fontsize=10,fmt='%d kPa',manual=mlocs)



# Plot operational range for N(t)
plt.plot([15]*2,[210,490],linewidth=4,color='orange',zorder=9,alpha=0.75)
plt.plot(15,350,'s',color='orange',markersize=14,alpha=0.75)
## This isn't quite right because the bed is kambered. Leave out for now
# plt.fill_between([7.5,22.5],[210]*2,[490]*2,color='orange',alpha=0.5,zorder=8)
plt.text(0.25,800,'No Cavities\n($S$ = 1)',fontsize=10,va='center')

# Plot V < V_{min} zone
plt.fill_between([0,4],[100]*2,[900]*2,color='black',alpha=0.1)

# Window dressing
plt.xlabel('Linear Velocity ($V$) [m a$^{-1}$]')
plt.ylabel('Effective Pressure ($N$) [ kPa ]')
plt.xlim([0,30])


if issave:
	OFILE = os.path.join(ODIR,'JGLAC_Fig02_Steady_State_Parameter_Space_%ddpi.%s'%(DPI,FMT.lower()))
	plt.savefig(OFILE,dpi=DPI,format=FMT.lower())


plt.show()

