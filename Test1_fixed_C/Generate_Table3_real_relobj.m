%%
%% Replicate the numerical results from Table 3
%%
clc;clear; % profile on
HOME = pwd; addpath(genpath(HOME));

filepath = fileparts(HOME);
datadir =  [filepath,filesep,'Data\Real_data'];


%% Results of sGS_isPADMM on real data
result_sGS = Test_sGS_isPADMM_real(1,[1 10],[1e-4 1e-6],datadir); 
%save result_sGS.mat result_sGS

%% Results of isPADMM on real data
result_isp = Test_isPADMM_real(1,[1 10],[1e-4 1e-6],datadir);
%save result_isp.mat result_isp

%% Results of F-ADMM on real data
result_fadmm = Test_FADMM_real(1,[1 10],[1e-4 1e-6],datadir);
%save result_fadmm.mat result_fadmm


result_table3 = zeros(16,15);
result_table3(:,[1:4 7 10 13]) = result_sGS(:,4:10);
result_table3(:,[6:3:15]) = result_isp(:,[8 10 9 11]);
result_table3(:,[5:3:14]) = result_fadmm(:,[8 10 9 11]);

save result_table3.mat result_table3










