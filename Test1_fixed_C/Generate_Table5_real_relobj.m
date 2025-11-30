%%
%% Replicate the numerical results from Table 5
%%
clc;clear; 
HOME = pwd; addpath(genpath(HOME));

filepath = fileparts(HOME);
datadir =  [filepath,filesep,'Data\Real_data'];

%% Results of ALMSNCG on random data
result_real_alm = Test_ALMSNCG_real([1:4],[1 10],[1e-4 1e-6],1,datadir); 
%save result_real_alm.mat result_real_alm 

%% Results of isPADMM on random data
result_real_isp = Test_isPADMM_real([1:4],[1 10],[1e-4 1e-6],datadir);
%save result_real_isp.mat result_real_isp

%% Results of F-ADMM on random data
result_real_fadmm = Test_FADMM_real([1:4],[1 10],[1e-4 1e-6],datadir);
%save result_real_fadmm.mat result_real_fadmm

result_table5 = zeros(48,24);
result_table5(:,[1:8 11 14 17 18 21 24]) = result_real_alm(:,[1:8 9 10 12 13 17 11]);
result_table5(:,[10 13 16 20 23]) = result_real_isp(:,[7 8 10 13 9]);
result_table5(:,[9 12 15 19 22]) = result_real_fadmm(:,[7 8 10 13 9]);

save result_table5.mat result_table5










