"""
:module: Model_UW_CRSD_Parameter_Space.py
:purpose: Calculate steady-state sliding theory values to predict cavity geometry
		  and drag (\\tau) values for a given parameter space of effective pressures
		  and slip velocities (N,V)

:auth: Nathan T. Stevens
:email: ntstevens@wisc.edu

"""
import os, logging, argparse
import sys
import numpy as np
import pandas as pd
sys.path.append(os.path.join('.'))
import lliboutry_kamb_model as lkm

Logger = logging.getLogger('Model_UW_CRSD_Parameter_Space')

def main():
	parser = argparse.ArgumentParser(
		
	)
	# Map directory structure
	ROOT = os.path.join('..','..','..')
	OROOT = os.path.join(ROOT,'processed','model')

	# Get UW-CRSD sinusoidal bed standard geometry
	bed_geom = lkm.defined_bed('UW')
	# Create parameter space basis vectors
	Nv = np.linspace(0.1e6,0.9e6,4001) #[Pa] Effective pressure range
	Vv = np.linspace(0,60,2001) #[m a^{-1}] Slip velocity range
	# Create holder for model outputs
	mod_mats = {'T_Pa':[],'N_Pa':[],'S':[]}
	# Iterate across velocity values
	for V_ in tqdm(Vv):
		# Update input velocity associated with `bed_geom`
		bed_geom.update({'US':V_})
		# Model 
		mod_row = lkm.calc_TS_vectors(Nv[0],Nv[-1],nnods=len(Nv),**bed_geom)
		for k_ in mod_row.keys():
			mod_mats[k_].append(mod_row[k_])

	print('Modeling done!')

	for k_ in mod_mats.keys():
		df = pd.DataFrame(np.array(mod_mats[k_]),columns=Nv,index=Vv)
		df.to_csv(os.path.join(OROOT,'model_results_%s.csv'%(k_)),header=True,index=True)


# # Create arrays for plotting
# Vm,Nm = np.meshgrid(Vv,Nv*1e-3)
# Tm = np.array(mod_mats['T_Pa']).T*1e-3
# Sm = np.array(mod_mats['S']).T


# # Initialize 
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cbh = ax.pcolor(Vm,Nm,Tm,cmap='Blues_r')
# Tch = ax.contour(Vm,Nm,Tm,levels=np.arange(0,400,50),colors='k')
# ax.contour(Vm,Nm,Tm,levels=[275],colors='r',linestyles='--')
# Sch = ax.contour(Vm,Nm,Sm,levels=np.arange(0,1.1,0.1),colors='w',linestyles=':')
# plt.colorbar(cbh)

# ax.clabel(Tch,Tch.levels,inline=True)
# ax.clabel(Sch,Sch.levels,inline=True)

# ax.set_ylabel('Effective pressure [$N$] (kPa)')
# ax.set_xlabel('Linear slip velocity [$V$] (m yr$^{-1}$)')


# plt.show()

# ax.contour(Vm,Nm*1e-3,np.array(mod_mats['T_Pa']))