import os, sys, logging, argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.datetimeindex import *

Logger = logging.getLogger('process_cavity_pick_data.py')

#### SUBROUTINES ####
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

def deg2rad(deg):
	return deg*(np.pi/180.)

def make_R2_mat(theta):
	RR = np.array([[np.cos(theta), - np.sin(theta)],\
				   [np.sin(theta),   np.cos(theta)]])
	RR = RR.reshape(2,2)
	return RR

def crest_rotate(S_data,S_corr):
	# Grab crest coordinate as rotation pole 
	x0,y0 = S_data['Crest X'],S_data['Crest Y']
	# Grab stoss and lee coordinate data
	xd = S_data[['Stoss X','Lee X']].values
	yd = S_data[['Stoss Y','Lee Y']].values
	# Perform prescribed rotation of stoss/lee coordinates
	rd = np.matmul(make_R2_mat(deg2rad(S_corr['thetaCC'])),np.c_[xd-x0,yd-y0].T)
	return rd


def y2xloc(rd,S_corr,lbda=0.25*np.pi*2*.3,aa=0.039):
	# Extract rotated vector coordinates
	xr,yr = rd[0,:].astype(float),rd[1,:].astype(float)
	# Apply scaling factor to X
	X = xr * S_corr['scalingX'].values[0]
	# Apply scalilng factor to Y and have the crest a height 2*amplitude
	Y = yr * S_corr['scalingY'].values[0] + aa*2
	# Use inverse of cosine bed model y = aa*(cos(kk*x) + 1)
	# to find the X-position based on 
	kk = np.pi*2./lbda
	X_hat = np.arccos((Y/aa) - 1)/kk

	return X_hat, X, Y

def process_geometry(S_data,S_corr,lbda=0.25*np.pi*2*.3,aa=0.039):
	rd = crest_rotate(S_data,S_corr)
	# breakpoint()
	X_hat, X, Y = y2xloc(rd,S_corr,lbda=lbda,aa=aa)
	# Get fractional position
	S_hat = X_hat/lbda
	return S_hat, X_hat, rd, X, Y


def main(args):
	
    if isinstance(args.input_file, str):
        pass
    else:
        Logger.critical('Invalid input file - quitting')
        sys.exit(1)
		
    # Organize picks into organized sets w/ DateTime indexing
    df_main = reorg_picks(pd.read_csv(args.input_file,parse_dates=['DateTime'],index_col='DateTime'))

    ## Put in camera indexing
    # Filter data to specified T24 experiment timestamps bounding Camera 4 imaging
    IND_c4 = (df_main.index >= pd.Timestamp("2021-10-28T12")) & (df_main.index <= pd.Timestamp("2021-10-30T13"))
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


    ## Calculate raw stoss & lee lengths for each camera
    df_stoss = pd.concat([df_main[['Stoss X','Stoss Y','Crest X','Crest Y','cam']],\
                                pd.Series(df_main['Crest X'] - df_main['Stoss X'],name='Stoss dx'),\
                                pd.Series(df_main['Crest Y'] - df_main['Stoss Y'],name='Stoss dy')],\
                                axis=1,ignore_index=False)
    df_lee = pd.concat([df_main[['Lee X','Lee Y','Crest X','Crest Y','cam']],\
                                pd.Series(df_main['Lee X'] - df_main['Crest X'],name='Lee dx'),\
                                pd.Series(df_main['Lee Y'] - df_main['Crest Y'],name='Lee dy')],\
                                axis=1,ignore_index=False)

    #### ADDITIONAL FINE-SCALE ADJUSTMENTS ####
    """
    Observations from extended viewing of time-lapse images and camera transfers
    1) Camera 4's vantage point provides a greater about of fitting surfaces to the flattened model
    so preference should be given to the general scaling of stoss contact areas for absolute values
    2) The transition from cam2 to cam4 a the subsequent transitions occurs when cavity dilation is at maximum, 
    which generally stabilizes cavity geometries. As such, the values of stoss & lee contact areas should
    match across these transitions
    3) Stoss contact areas are observed to be slightly larger for the first peak observed by cam2 (c. 2021-10-28) 
    compared to the second peak (c. 2021-10-31)
    4) Stoss contact areas are observed to be slightly smaller for the first peak observed by cam4 (c. 2021-10-29)
    compared to the second peak (c. 2021-10-30)
    5) Due to the proximity of detachment points (lee contact areas) compared to reattachement points (stoss contact
    areas) lee contact length estimates are likely less sensitive to re-projection artifacts.

    """


    # Fetch stoss length values at each side of camera transfers
    for i_ in range(len(IND_c4) - 1):
        if not IND_c4[i_] and IND_c4[i_ + 1]:
            loc1 = [i_, i_ + 1]
        elif IND_c4[i_] and not IND_c4[i_ + 1]:
            loc2 = [i_, i_ + 1]


    ## Extract distance data
    # Get Stoss delta X values
    S_dxs = df_stoss['Stoss dx']
    S_dys = df_stoss['Stoss dy']
    L_dxs = df_lee['Lee dx']
    L_dys = df_lee['Lee dy']

    ## Process dX data first
    # Extract reference locations for X coordinates
    # reference from 
    sxref0 = S_dxs.values[loc1[0]]
    sxref1 = S_dxs.values[loc1[1]]
    sxref2 = S_dxs.values[loc2[0]]
    sxref3 = S_dxs.values[loc2[1]]

    syref0 = S_dys.values[loc1[0]]
    syref1 = S_dys.values[loc1[1]]
    syref2 = S_dys.values[loc2[0]]
    syref3 = S_dys.values[loc2[1]]

    S2_dxs = S_dxs[~IND_c4]
    S2_dys = S_dys[~IND_c4]
    S4_dxs = S_dxs[IND_c4]
    S4_dys = S_dys[IND_c4]

    # u4 = S4_dxs.mean()
    # u2b = np.mean([sxref0,sxref3])


    lxref0 = L_dxs.values[loc1[0]]
    lxref1 = L_dxs.values[loc1[1]]
    lxref2 = L_dxs.values[loc2[0]]
    lxref3 = L_dxs.values[loc2[1]]

    lyref0 = L_dys.values[loc1[0]]
    lyref1 = L_dys.values[loc1[1]]
    lyref2 = L_dys.values[loc2[0]]
    lyref3 = L_dys.values[loc2[1]]

    L2_dxs = L_dxs[~IND_c4]
    L2_dys = L_dys[~IND_c4]
    L4_dxs = L_dxs[IND_c4]
    L4_dys = L_dys[IND_c4]

    # L2_dxs = L_dxs[~IND_c4]
    # L4_dxs = L_dxs[IND_c4]


    ##### START WITH CAM2 CORRECTIONS BASED ON POINTS 1--3 FROM ABOVE #####
    # More likely scenario because we can get more definitive features from cam4's coverage
    # i.e., 1 crest, 1 trough, 1 full stoss face, and 2 partial lee faces
    # camera 2 only shows a crest and halves of 2 faces

    if args.show_plots:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(S_dxs,'.',label="Crest Corrected")

    # Shift cam2 data to line up at first transition
    # This transition corresponds with a period of max dilation when
    # contact areas should be relatively low and stable
    DC = sxref1 - sxref0 
    # DC = u4 - u2b

    ## Apply DC correction to S2 data to match average values
    S_dxsc1 = pd.concat([S4_dxs,S2_dxs + DC],axis=0).sort_index()
    S_dxsc1.name = 'Stoss X DC'
    if args.show_plots:
        ax1.plot(S_dxsc1,'o',label='Offset corrected')

    ##### NOW CORRECT CAM4 BASED ON POINTS 1--4 FROM ABOVE #####
    # Get the slope of cam4 data - this assumes a linear time-dependent adjustment
    # While somewhat nonphysical, this is a simpler model for correcting artifacts from
    # the nonlinear image projection onto a focal-plane used

    # Get slope of average linear fit to cam4 stoss contact lengths
    mod = fit_dateline(S4_dxs.index,S4_dxs.values)
    # Define any scalar modifications here and print out model line being subtracted
    # from cam4 stoss contact length observations
    mod *= 1
    ref_v = sxref1
    print('Correction model for camera 4 data: %.3e t + %.3e '%(mod,ref_v))
    # Reduce data by the defined linear model
    S4_dxstc = reduce_series_by_slope(S4_dxs,mod*1.2,S4_dxs.index.min(),sxref1)

    # Recombine into fully corrected series
    S_dxsc2 = pd.concat([S4_dxstc + ref_v,S2_dxs + DC],axis=0).sort_index()
    S_dxsc2.name = 'Stoss X DC+M'
    if args.show_plots:
        ax1.plot(S_dxsc2,'v-',label='Offset + C4 Slope Correction',alpha=0.5)

    if args.show_plots:
        ax1b = ax1.twinx()
        ax1b.plot(df_lee['Lee X'],'ro-',alpha=0.5)




    ##### ALTERNATE APPROACH - PIECEWISE STITCHGING #####
    # Hypothesis is that mismatches in edge fits arise from some adjustment between cam2 
    # scene 1 and cam2 scene 2
    # This seeks to preserve 

    ### STOSS PROCESSING ####
    # dref0 = S2_dxs[S2_dxs.index >= S_dxs.index[loc1[0]] - pd.Timedelta(6,unit='hour')].median()
    # dref1 = S4_dxs[S4_dxs.index <= S_dxs.index[loc1[1]] + pd.Timedelta(6,unit='hour')].median()
    # dref2 = S4_dxs[S4_dxs.index >= S_dxs.index[loc2[0]] - pd.Timedelta(48,unit='hour')].median()
    # dref3 = S2_dxs[S2_dxs.index <= S_dxs.index[loc2[1]] - pd.Timedelta(24,unit='hour')].median()


    ### STOSS PROCESSING ###
    # Calculate camera 2 horizontal corrections 
    DXC1 = sxref0 - sxref1
    DXC2 = sxref3 - sxref2
    # Calculate camera 2 vertical corrections
    DYC1 = syref0 - syref1
    DYC2 = syref3 - syref2

    ## Create offset-corrected Camera 2 Stoss elements
    # X-components
    S_c2s1_dxs = S_dxs[S_dxs.index <= S_dxs.index[loc1[0]]] - DXC1
    S_c2s2_dxs = S_dxs[S_dxs.index >= S_dxs.index[loc2[1]]] - DXC2
    # Y-components
    S_c2s1_dys = S_dys[S_dys.index <= S_dys.index[loc1[0]]] - DYC1
    S_c2s2_dys = S_dys[S_dys.index >= S_dys.index[loc2[1]]] - DYC2

    ## Put together series with Camera 4 observations
    Sx_DC = pd.concat([S_c2s1_dxs,S4_dxs,S_c2s2_dxs],axis=0).sort_index()
    Sx_DC.name = 'Stoss X Piecewise'

    Sy_DC = pd.concat([S_c2s1_dys,S4_dys,S_c2s2_dys],axis=0).sort_index()
    Sy_DC.name = 'Stoss Y Piecewise'

    if args.show_plots:
        ax1.plot(Sx_DC,'k^',label='24-mean edge offsets')

    ### LEE PROCESSING ###
    # Calculate camera 2 vertical corrections
    DXC1 = lxref0 - lxref1
    DXC2 = lxref3 - lxref2
    # Calculate camera 2 vertical corrections
    DYC1 = lyref0 - lyref1
    DYC2 = lyref3 - lyref2

    ### Create offset-corrected Camera 2 Lee elements
    # X-components
    L_c2s1_dxs = L_dxs[L_dxs.index <= L_dxs.index[loc1[0]]] - DXC1
    L_c2s2_dxs = L_dxs[L_dxs.index >= L_dxs.index[loc2[1]]] - DXC2

    # Y-components
    L_c2s1_dys = L_dys[L_dys.index <= L_dys.index[loc1[0]]] - DYC1
    L_c2s2_dys = L_dys[L_dys.index >= L_dys.index[loc2[1]]] - DYC2

    Lx_DC = pd.concat([L_c2s1_dxs,L4_dxs,L_c2s2_dxs],axis=0).sort_index()
    Lx_DC.name = 'Lee X Piecewise'

    Ly_DC = pd.concat([L_c2s1_dys,L4_dys,L_c2s2_dys],axis=0).sort_index()
    Ly_DC.name = 'Lee Y Piecewise'

    if args.show_plots:
        ax1.set_xlabel("Date Time (UTC - 5)")
        ax1.set_ylabel("Stoss contact length (m)")
        ax1b.set_ylabel("Lee contact length (m)",color='red')
        ax1.legend()

        plt.show()





    df_out = pd.concat([df_lee['Lee X'],df_lee['Lee Y'],df_stoss['Stoss X'],df_stoss['Stoss Y'],\
                            Lx_DC,Ly_DC,Sx_DC,Sy_DC,S_dxs,S_dxsc1,S_dxsc2],axis=1,ignore_index=False)
    Logger.info(f'Applying timezone correction of {args.tz} hours to data timestamps to correct to UTC')
    df_out.index -= pd.Timedelta(args.tz, unit='hour')
    df_out = df_out.assign(Epoch_UTC=[x.timestamp() for x in df_out.index])
    # 
    # df_out.to_csv(os.path.join(ODIR,'Postprocessed_Contact_Areas.csv'),header=True,index=True)
    Logger.info(f'Writing to disk: {args.output_file}')
    df_out.to_csv(args.output_file, header=True, index=False)




if __name__ == '__main__':
    ch = logging.StreamHandler()                                                            
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(fmt)
    Logger.addHandler(ch)
    Logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        prog='process_cavity_pick_data.py',
        description='conduct minor projection corrections on extracted cavity geometry data'
    )

    parser.add_argument(
        '-i',
        '--input',
        dest='input_file',
        action='store',
        default=None,
        help='target *.csv file with raw cavity attachment/detachment point position measurements',
        type=str
    )

    parser.add_argument(
		'-o',
		'--output',
		dest='output_file',
		action='store',
		default='./tmp_processed_cavity_picks.csv',
        help='file to write cleaned cavity picks to (include *.csv extension)',
        type=str
    )

    parser.add_argument(
          '-s',
          '-show_plots',
          action='store_true',
          dest='show_plots',
    )

    parser.add_argument(
         '-z',
         '--timezone_correction',
         action='store',
         dest='tz',
         default=-5.,
         type=float
    )

    args = parser.parse_args()


    main(args)
