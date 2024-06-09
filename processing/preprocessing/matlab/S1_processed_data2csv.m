% S1_processed_data2csv.m - convert processed vertical load (N),
% 							shear stress (Tau) and water pressure (P)
%							data from S0_RS_dataConversion_OSC_N.m
%							into *.csv files
%
%:auth: Nathan T. Stevens
%:email: ntstevens@wisc.edu

clear all;
close all;
% Data Paths

NTPDATA = '../../../data/NTauP/RS_procData_OSC_N.mat'
ODIR = '../../../data/NTauP'

%% Load N,T,and P dataData
load(NTPDATA);

%% Compose tables
N_kPa = SigmaN;
T_kPa = Tau;
Pw1_kPa = Pw1;
Pw2_kPa = Pw2;
% Do UTC Time indexed data first
t_NTP_UTC = table(TimeUTC,N_kPa,T_kPa,Pw1_kPa,Pw2_kPa);
% then do (partial) local time indexed data
t_NT_LOC = table(LocalTime_stress,Sigma_kPa,Tau_kPa);

%% Write tables to CSV
writetable(t_NTP_UTC,[ODIR,'/UTC_N_T_P.csv']);
writetable(t_NT_LOC,[ODIR,'/LOC_N_T.csv']);
