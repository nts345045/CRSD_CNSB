"""
:module: S2_cavity_geometry_modeling.py
:purpose: Apply a range of analytic techniques to seek a model wherein
		 ice-bed separation (dZ) reasonably predicts ice-bed contact length (S).

		 S = f(dZ,...)

		 considering a number of definitions of S that are informed by direct obs

		 OR

		 Truly, we seek a predictive model wherein ice-bed separation variations
		 

"""
import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.join('..'))
from util.datetimeindex import pick_extrema_indices
import core.model.steadystate_model as ssm

# Map Directories
ROOT = os.path.join('..','..','..')
DDIR = os.path.join(ROOT,'processed','timeseries')
# Map Experimental Data Files from T24
T24_NT = os.path.join(DDIR,'S3_T24_NT_10min_x3_smoothed.csv')
T24_LV = os.path.join(DDIR,'S4_T24_LVDT_10min_x3_smoothed_QSS_reduced.csv')
T24_SM = os.path.join(ROOT,'processed','Cavity_Picks','Postprocessed_Contact_Geometries.csv')

# Map Experimental Data Files from T6
T6_NT = os.path.join(DDIR,'S3_T06_NT_10min_x3_smoothed.csv')
T6_LV = os.path.join(DDIR,'S4_T06_LVDT_10min_x3_smoothed_QSS_reduced.csv')


# LOAD EXPERIMENTAL DATA #
df_NT24 = pd.read_csv(T24_NT,parse_dates=True,index_col=[0])
df_Z24 = pd.read_csv(T24_LV,parse_dates=True,index_col=[0])
df_S24 = pd.read_csv(T24_SM,parse_dates=True,index_col=[0])
df_NT6 = pd.read_csv(T6_NT,parse_dates=True,index_col=[0])
df_Z6 = pd.read_csv(T6_LV,parse_dates=True,index_col=[0])

df_Z6 = df_Z6.resample(pd.Timedelta(3,unit='sec')).interpolate(method='from_derivatives')
# SET REFERENCES/CONSTANTS
hh_cl = 25.3*2  # [mm] Centerline bump height
lbda_cl = 314.25
hh_ow = 39.*2   # [mm] Outer experimental chamber bump height
lbda_ow = 0.47123889803846897e3
# Get offset models from observed reattachment and detachment point heights
dRstoss = df_S24['Stoss dY POR mm'].values[-1]/hh_ow
dRlee = df_S24['Lee dY PRO mm'].values[-1]/hh_ow
dRmea = np.mean([dRstoss,dRlee])


# Experiment Starting Times
t0_T24 = pd.Timestamp('2021-10-26T13:58')
t0_T6 = pd.Timestamp('2021-11-1T11:09:15')

### CALCULATE STEADY-STATE VALUES ASSUMING N(t) FORCING IS THE ONLY FREE VARIABLE
# WITH EVERYTHING ELSE COMING FROM BED/CHAMBER GEOMETRY AND ZOET & IVERSON (2015)
# EFFECTIVE VISCOSITY
# Populate \tau(N) and S(N) vectors from steady-state model
model_vectors = ssm.calc_TS_vectors(100e3,600e3,nnods=10001)
# Populate estimates of cavity roof height
SR_vectors = ssm.calc_SR_vectors(100e3,600e3,nnods=10001,npts=10001)



dict_out = model_vectors.copy()
dict_out.update(SR_vectors)
dict_out = {k_:dict_out[k_] for k_ in set(list(dict_out.keys())) - set(['N_Pa'])}
df_MOD = pd.DataFrame(dict_out,index=pd.Index(model_vectors['N_Pa'],name='N_Pa'))

### CONDUCT PROCESSING FOR T24 GEOMETRIES

# Downsample N(t) and \tau(t) data to match sampling from LVDT data
NPa_ds_T24 = np.interp((df_Z24.index - t0_T24).total_seconds(),\
					(df_NT24.index - t0_T24).total_seconds(),\
					df_NT24['N_kPa'].values)*1e3
TPa_ds_T24 = np.interp((df_Z24.index - t0_T24).total_seconds(),\
					(df_NT24.index - t0_T24).total_seconds(),\
					df_NT24['T_kPa'].values)*1e3

## Get reference \\Delta y(t)
IND = np.argmax(np.abs((df_Z24.index - t0_T24).total_seconds()))
## Calculate Reattachment Point Height Steady-State Reference Value
DR0 = df_Z24['LVDT_mm_stitched red'].values[IND]/hh_cl



# INTERPOLATE STEADY-STATE VALUES TO N(t) INPUTS
T_hat24 = np.interp(NPa_ds_T24,model_vectors['N_Pa'],model_vectors['T_Pa'])
S_hat24 = np.interp(NPa_ds_T24,model_vectors['N_Pa'],model_vectors['S'])

# Calculate non-dimensionalized cavity height & contact length metrics
# Assume mean of cavity height is representative for ice-bed separation
R_mea_hat24 = np.interp(NPa_ds_T24,model_vectors['N_Pa'],SR_vectors['Rmea'])
S_tot_hat24 = np.interp(NPa_ds_T24,model_vectors['N_Pa'],SR_vectors['Stot'])
# Assume minimum cavity height (reattachment point) is representative for ice-bed separation
R_stoss_hat24 = np.interp(NPa_ds_T24,model_vectors['N_Pa'],SR_vectors['Rstoss'])
S_stoss_hat24 = np.interp(NPa_ds_T24,model_vectors['N_Pa'],SR_vectors['Sstoss'])
# Assume maximum cavity height (detachment point) is representative for ice-bed separation
R_lee_hat24 = np.interp(NPa_ds_T24,model_vectors['N_Pa'],SR_vectors['Rlee'])
S_lee_hat24 = np.interp(NPa_ds_T24,model_vectors['N_Pa'],SR_vectors['Slee'])



### Calculate observed cavity height metrics from LVDT + observed SS 
# Calculate R(t) from stoss (reattachment) point
Rstoss = (df_Z24['LVDT_mm_stitched red'].values/hh_cl) - DR0 + (1 + dRstoss)
Sstoss = np.interp(Rstoss,SR_vectors['Rstoss'],SR_vectors['Sstoss'],period=24)
# Calculate R(t) from lee (detachment) point
Rlee = (df_Z24['LVDT_mm_stitched red'].values/hh_cl) - DR0 + (1 + dRlee)
Slee = np.interp(Rlee,SR_vectors['Rlee'],SR_vectors['Slee'],period=24)
# Calculate R(t) from average of lee and stoss points
Rbar = (df_Z24['LVDT_mm_stitched red'].values/hh_cl) - DR0 +(1 + dRmea)
Stot = np.interp(Rbar,SR_vectors['Rmea'],SR_vectors['Stot'],period=24)

# CALCULATE DRAG FROM OBSERVED N(t) AND R(t) ASSUMING THE R2S MAPPING FROM STEADY STATE THEORY IS CORRECT
# R2S_hat = np.interp(Rlee,df_MOD['Rmea'].values,df_MOD['S'].values)
# R2S2tau_lee_hat = ssm.calc_from_SN(NPa_ds_T24,R2S_hat)
# R2Sstoss_hat = np.interp(Rstoss,df_MOD['Rstoss'].values,df_MOD['Sstoss'].values)
# R2Stot_hat = np.interp(Rbar,df_MOD['Rmea'].values,df_MOD['S'].values)


df_summary_T24 = pd.DataFrame({'N kPa':NPa_ds_T24*1e-3,\
							   'T kPa':TPa_ds_T24*1e-3,'hat T kPa':T_hat24*1e-3,\
							   'R stoss':Rstoss,'hat R stoss':R_stoss_hat24,\
							   'R mea':Rbar,'hat R mea':R_mea_hat24,\
							   'R lee':Rlee,'hat R lee':R_lee_hat24,\
							   'S tot':Stot,'hat S tot':S_tot_hat24,\
							   'S lee':Slee,'hat S lee':S_lee_hat24,\
							   'S stoss':Sstoss,'hat S stoss':S_stoss_hat24},index=df_Z24.index)



## CONDUCT PROCESSING FOR T6 GEOMETRIES

NPa_ds_T6 = np.interp((df_Z6.index - t0_T6).total_seconds(),\
					(df_NT6.index - t0_T6).total_seconds(),\
					df_NT6['N_kPa'].values)*1e3
TPa_ds_T6 = np.interp((df_Z6.index - t0_T6).total_seconds(),\
					(df_NT6.index - t0_T6).total_seconds(),\
					df_NT6['T_kPa'].values)*1e3

## Get reference \\Delta y(t)
IND = np.argmin(np.abs((df_Z6.index - t0_T6).total_seconds()))
## Calculate Reattachment Point Height Steady-State Reference Value
DR0 = df_Z6['LVDT_mm_stitched red'].values[IND]/hh_cl

# INTERPOLATE STEADY-STATE VALUES TO N(t) INPUTS
T_hat6 = np.interp(NPa_ds_T6,model_vectors['N_Pa'],model_vectors['T_Pa'])
S_hat6 = np.interp(NPa_ds_T6,model_vectors['N_Pa'],model_vectors['S'])

# Calculate non-dimensionalized cavity height for a range of models
# Assume mean of cavity height is representative for ice-bed separation
R_mea_hat6 = np.interp(NPa_ds_T6,model_vectors['N_Pa'],SR_vectors['Rmea'])
S_tot_hat6 = np.interp(NPa_ds_T6,model_vectors['N_Pa'],SR_vectors['Stot'])
# Assume minimum cavity height (reattachment point) is representative for ice-bed separation
R_stoss_hat6 = np.interp(NPa_ds_T6,model_vectors['N_Pa'],SR_vectors['Rstoss'])
S_stoss_hat6 = np.interp(NPa_ds_T6,model_vectors['N_Pa'],SR_vectors['Sstoss'])
# Assume maximum cavity height (detachment point) is representative for ice-bed separation
R_lee_hat6 = np.interp(NPa_ds_T6,model_vectors['N_Pa'],SR_vectors['Rlee'])
S_lee_hat6 = np.interp(NPa_ds_T6,model_vectors['N_Pa'],SR_vectors['Slee'])


# Calculate Observed R(t) for a range of statistical representations
# Calculate R(t) from stoss (reattachment) point
Rstoss_T6 = (df_Z6['LVDT_mm_stitched red'].values/hh_cl) - DR0 + (1 + dRstoss)
Sstoss_T6 = np.interp(Rstoss_T6,SR_vectors['Rstoss'],SR_vectors['Sstoss'],period=6)
# Calculate R(t) from lee (detachment) point
Rlee_T6 = (df_Z6['LVDT_mm_stitched red'].values/hh_cl) - DR0 + (1 + dRlee)
Slee_T6 = np.interp(Rlee_T6,SR_vectors['Rlee'],SR_vectors['Slee'],period=6)
# Calculate R(t) from average of lee and stoss points
Rbar_T6 = (df_Z6['LVDT_mm_stitched red'].values/hh_cl) - DR0 + (1 + dRmea)
Stot_T6 = np.interp(Rbar_T6,SR_vectors['Rmea'],SR_vectors['Stot'],period=6)

df_summary_T6 = pd.DataFrame({'N kPa':NPa_ds_T6*1e-3,'T kPa':TPa_ds_T6*1e-3,'hat T kPa':T_hat6*1e-3,\
							   'R stoss':Rstoss_T6,'hat R stoss':R_stoss_hat6,\
							   'R mea':Rbar_T6,'hat R mea':R_mea_hat6,\
							   'R lee':Rlee_T6,'hat R lee':R_lee_hat6,\
							   'S tot':Stot_T6,'hat S tot':S_tot_hat6,\
							   'S lee':Slee_T6,'hat S lee':S_lee_hat6,\
							   'S stoss':Sstoss_T6,'hat S stoss':S_stoss_hat6},index=df_Z6.index)



## Write Corrected & Modeled Heights to File
R6_out = os.path.join(DDIR,'S5_experiment_T6_cavity_metrics.csv')
R24_out = os.path.join(DDIR,'S5_experiment_T24_cavity_metrics.csv')
Rmod_out = os.path.join(DDIR,'S5_modeled_values.csv')
df_summary_T6.to_csv(R6_out,header=True,index=True)
df_summary_T24.to_csv(R24_out,header=True,index=True)
df_MOD.to_csv(Rmod_out,header=True,index=True)
