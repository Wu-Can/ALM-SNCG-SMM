%%
%% Replicate the numerical results from Table 4
%%
clc;clear; 
%profile on
HOME = pwd; addpath(genpath(HOME));

filepath = fileparts(HOME);
datadir =  [filepath,filesep,'Data\Random_data'];

%% Results of ALM-SNCG on random data
result_random_alm = Test_ALMSNCG_random([1:4],[10 100],[1e-4, 1e-6],1,datadir);                                                                      
%save result_random_alm.mat result_alm 

%% Results of isPADMM on random data
result_random_isp = Test_isPADMM_random([1:4],[10 100],[1e-4, 1e-6],datadir); 
%save result_random_isp.mat result_isp

%% Results of F-ADMM on random data
result_random_fadmm = Test_FADMM_random([1:2],[10 100],[1e-4, 1e-6],datadir);
%save result_random_fadmm.mat result_fadmm

result_table6 = zeros(48,16);
result_table6(:,[1:8 11 14 17 18 21 24]) = result_random_alm(:,[1:8 9 10 12 13 17 11]);
result_table6(:,[10 13 16 20 23]) = result_random_isp(:,[7 8 10 13 9]);
result_table6(:,[9 12 15 19 22]) = result_random_fadmm(:,[7 8 10 13 9]);

save result_table6.mat result_table6











