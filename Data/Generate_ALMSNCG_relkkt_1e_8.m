%% Create benchmark solutions
clc;clear; % profile on
HOME = pwd; addpath(genpath(HOME));

%% Generate and save high-accuracy results as benchmarks on random data
datadir = [HOME,filesep,'Random_data'];

result = Test_ALMSNCG_random([1:4],[10 100],1e-8,0,datadir);
save result_ALMSNCG_random_relkkt_1e-08.mat result


%% Generate and save high-accuracy results as benchmarks on real data
datadir = [HOME,filesep,'Real_data'];

result = Test_ALMSNCG_real([1:4],[1 10],1e-8,0,datadir);
save result_ALMSNCG_real_relkkt_1e-08.mat result






