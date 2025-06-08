%%==========================================================================================
%% Test AS+ALM performance for the Support Matrix Machine (SMM) models 
%% with a sequence {C_i}^N_{i=1} using random data
%%
%% resultAvgAS = Test_AS_path_random(prob_vec,tau_vec,tol_vec,edgp,datadir)
%%
%% INPUT:
%% prob_vec = vector of problem orders
%% tau_vec = vector of tau parameter values in the SMM models
%% tol_vec = vector of ALM-SNCG tolerance
%% edgp = number of equally spaced grid points between the maximum and minimum values of C
%% datadir = path to the directory containing random data files
%==========================================================================================
function resultAvgAS = Test_AS_path_random(prob_vec,tau_vec,tol_vec,edgp,datadir)

C_vec = 10.^([-1:3/edgp:2]); 

%% ------------------------- input X and y ----------------------
filepath = fileparts(fileparts(datadir));
datadir2 =  [filepath,filesep,'Test2_path_Cvec'];
pathname = [datadir2,filesep,'Result_solution_path_figure_random',filesep,...
    'Result_AS_path_random',filesep];
if ~exist(pathname,'dir'), mkdir(pathname); end
addpath(genpath(pathname));

lenqq = length(prob_vec);
lentau = length(tau_vec); lentol = length(tol_vec);
lenC = length(C_vec); rr = 14;
resulthistAS = zeros(lenqq*lentau*rr,lenC);
resultAvgAS = zeros(lenqq*lentau,rr);
%%
for qq = 1:lenqq
    qqq = prob_vec(qq);
    switch qqq
        case 1
            n = 1e4; p = 100; q = 100;
        case 2
            n = 1e4; p = 1000; q = 500;
        case 3
            n = 1e5; p = 50; q = 100;
        case 4
            n = 1e6; p = 50; q = 100;
    end

    %% Generate Xy_train and Xy_test
    eval(['prob_test = [datadir,filesep,''Xy_test_',num2str(n),'_',num2str(p),'_',num2str(q),'.mat''];']);
    eval(['prob_train = [datadir,filesep,''Xy_train_',num2str(n),'_',num2str(p),'_',num2str(q),'.mat''];']);
    load(prob_test);
    load(prob_train);

    OPTIONS.n = n;
    OPTIONS.p = p;
    OPTIONS.q = q;
    OPTIONS.flag_scaling = 0;
    OPTIONS.sigmaiter = 2;
    OPTIONS.sigmascale = 1.2;
    OPTIONS.Test_sigma = 1;
    OPTIONS.C_vec = C_vec;
    OPTIONS.tol_nSM = -1e-12;
    OPTIONS.ifrandom = 1;

    resultAS_nSMM_path = zeros(lentau,lenC);
    resultAS_nImean_path = zeros(lentau,lenC);
    for oo = 1:lentol
        OPTIONS.tol = tol_vec(oo);
        for tt = 1:lentau
            OPTIONS.tau = tau_vec(tt);
            if n >= p*q
                num_sig = 100;
                max_sig = 3e3;
                OPTIONS.sigma = min(num_sig*OPTIONS.C_vec,max_sig);
            else
                OPTIONS.sigma = 100*ones(lenC,1);
            end

            [objopt,~,~,runhist,~] = AS_ALM(Xy_train.X,Xy_train.y,OPTIONS);

            resultAS_nSMM_path(tt+(qq-1)*lentau,:) = runhist.nSMM_path;
            resultAS_nImean_path(tt+(qq-1)*lentau,:) = runhist.nI_mean;

            resulthistAS(1+(tt-1)*rr+(qq-1)*lentau*rr,:) =  OPTIONS.C_vec;
            resulthistAS(2+(tt-1)*rr+(qq-1)*lentau*rr,:) =  OPTIONS.tau;
            resulthistAS(3+(tt-1)*rr+(qq-1)*lentau*rr,:) =  runhist.sigma;
            resulthistAS(4+(tt-1)*rr+(qq-1)*lentau*rr,:) = runhist.nI_mean;
            resulthistAS(5+(tt-1)*rr+(qq-1)*lentau*rr,:) = runhist.nSMM_path;
            resulthistAS(6+(tt-1)*rr+(qq-1)*lentau*rr,:) =  runhist.r_indexJ_ALM;
            resulthistAS(7+(tt-1)*rr+(qq-1)*lentau*rr,:) =  runhist.relkkt;
            resulthistAS(8+(tt-1)*rr+(qq-1)*lentau*rr,:) =  runhist.relgap;
            resulthistAS(9+(tt-1)*rr+(qq-1)*lentau*rr,:) =  runhist.iterAS;
            resulthistAS(10+(tt-1)*rr+(qq-1)*lentau*rr,:) =  runhist.iter_ALM;
            resulthistAS(11+(tt-1)*rr+(qq-1)*lentau*rr,:) =  runhist.iter_SSN;
            resulthistAS(12+(tt-1)*rr+(qq-1)*lentau*rr,:) =  runhist.time;
            resulthistAS(13+(tt-1)*rr+(qq-1)*lentau*rr,:) = objopt(:,1)';
            resulthistAS(14+(tt-1)*rr+(qq-1)*lentau*rr,:) = runhist.ttime_path;

            resultAvg_AS_vec = mean(resulthistAS(1+(tt-1)*rr+(qq-1)*lentau*rr:14+(tt-1)*rr+(qq-1)*lentau*rr,2:end),2);

            resultAvgAS(tt+(oo-1)*lentau+(qq-1)*lentau*lentol,1) = OPTIONS.n;
            resultAvgAS(tt+(oo-1)*lentau+(qq-1)*lentau*lentol,2) = OPTIONS.p;
            resultAvgAS(tt+(oo-1)*lentau+(qq-1)*lentau*lentol,3) = OPTIONS.q;
            resultAvgAS(tt+(oo-1)*lentau+(qq-1)*lentau*lentol,4) = OPTIONS.tol;
            resultAvgAS(tt+(oo-1)*lentau+(qq-1)*lentau*lentol,5) = OPTIONS.tau;
            resultAvgAS(tt+(oo-1)*lentau+(qq-1)*lentau*lentol,6) = resultAvg_AS_vec(5); % Avg_nSM
            resultAvgAS(tt+(oo-1)*lentau+(qq-1)*lentau*lentol,7) = resultAvg_AS_vec(4); % Avg_sam
            resultAvgAS(tt+(oo-1)*lentau+(qq-1)*lentau*lentol,8) = max(resulthistAS(4+(tt-1)*rr+(qq-1)*lentau*rr,2:end)); % Max_sam;
            resultAvgAS(tt+(oo-1)*lentau+(qq-1)*lentau*lentol,9) = resultAvg_AS_vec(6); % Avg_|J_1|
            resultAvgAS(tt+(oo-1)*lentau+(qq-1)*lentau*lentol,10) =  max(resulthistAS(7+(tt-1)*rr+(qq-1)*lentau*rr,2:end)); % Worst relkkt
            resultAvgAS(tt+(oo-1)*lentau+(qq-1)*lentau*lentol,11) = resultAvg_AS_vec(9); % Avg_iterAS
            resultAvgAS(tt+(oo-1)*lentau+(qq-1)*lentau*lentol,12) = resultAvg_AS_vec(10); % Avg_iter_ALM
            resultAvgAS(tt+(oo-1)*lentau+(qq-1)*lentau*lentol,13) = resultAvg_AS_vec(11); % Avg_iter_SSN
            resultAvgAS(tt+(oo-1)*lentau+(qq-1)*lentau*lentol,14) = resultAvg_AS_vec(12); % Avg_time
        end
        eval(['filename_nSMM = ''resultAS_nSMM_random_path_',num2str(n),'_',num2str(p),'_',num2str(q),'_',num2str(edgp),'_',num2str(OPTIONS.tol),''';']);
        eval(['filename_nImean = ''resultAS_nImean_random_path_',num2str(n),'_',num2str(p),'_',num2str(q),'_',num2str(edgp),'_',num2str(OPTIONS.tol),''';']);
        
        save([pathname,filename_nSMM,'.mat'],'resultAS_nSMM_path');
        save([pathname,filename_nImean,'.mat'],'resultAS_nImean_path');
    end
end


