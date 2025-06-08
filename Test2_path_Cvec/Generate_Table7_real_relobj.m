%%
%% Replicate the numerical results from Table 7
%%
clc;clear; % profile on
HOME = pwd; addpath(genpath(HOME));

filepath = fileparts(HOME);
datadir =  [filepath,filesep,'Data\Real_data'];

%% Results of AS+ALM on real data
result_AS_real = Test_AS_path_real([1 2],[1 10],[1e-4 1e-6],50,datadir);
%save result_AS_real.mat result_AS_real


%% Results of Warm+ALM on real data
result_Warm_real = Test_Warm_path_real([1 2],[1 10],[1e-4 1e-6],50,datadir);
%save result_Warm_real.mat result_Warm_real

result_table7 = zeros(8,19);
result_table7(:,[1:8 10 12 15:17 19]) = result_AS_real;
result_table7(:,[9 11 13 14 18]) = result_Warm_real(:,6:10);

save result_table7.mat result_table7









