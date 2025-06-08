%%
%% Replicate the numerical results from Table 6
%%
clc;clear; % profile on
HOME = pwd; addpath(genpath(HOME));

filepath = fileparts(HOME);
datadir =  [filepath,filesep,'Data\Random_data'];

%% Results of AS+ALM on random data
result_AS_random = Test_AS_path_random([1:4],[10 100],[1e-4 1e-6],50,datadir);
%save result_AS_random.mat result_AS_random


%% Results of Warm+ALM on random data
result_Warm_random = Test_Warm_path_random([1:4],[10 100],[1e-4 1e-6],50,datadir);
%save result_Warm_random.mat result_Warm_random


result_table6 = zeros(16,19);
result_table6(:,[1:8 10 12 15:17 19]) = result_AS_random;
result_table6(:,[9 11 13 14 18]) = result_Warm_random(:,6:10);

save result_table6.mat result_table6









