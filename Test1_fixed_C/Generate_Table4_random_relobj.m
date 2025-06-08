%%
%% Replicate the numerical results from Table 4
%%
clc;clear; % profile on
HOME = pwd; addpath(genpath(HOME));

filepath = fileparts(HOME);
datadir =  [filepath,filesep,'Data\Random_data'];


%% Results of ALMSNCG on random data
result_alm = Test_ALMSNCG_random([1:4],[10 100],[1e-4 1e-6],1,datadir);                                                                        
%save result_alm.mat result_alm 

%% Results of isPADMM on random data
result_isp = Test_isPADMM_random([1:4],[10 100],[1e-4 1e-6],datadir);
%save result_isp.mat result_isp

%% Results of F-ADMM on random data
result_fadmm = Test_FADMM_random([1:2],[10 100],[1e-4 1e-6],datadir);
%save result_fadmm.mat result_fadmm

row_fadmm = size(result_fadmm,1);

result_table4 = zeros(64,17);
result_table4(:,[1:8,11,14,17]) = result_alm(:,[1:8,9:11]);
result_table4(:,[10,13,16]) = result_isp(:,7:9);
result_table4(1:row_fadmm,[9,12,15]) = result_fadmm(1:row_fadmm,7:9);

save result_table4.mat result_table4










