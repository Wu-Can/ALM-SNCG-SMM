%%==========================================================================================
%% Test Warm+ALM performance for the Support Matrix Machine (SMM) models 
%% with a sequence {C_i}^N_{i=1} using random data
%%
%% resultAvgWarm = Test_Warm_path_random(prob_vec,tau_vec,tol_vec,edgp,datadir)
%%
%% INPUT:
%% prob_vec = vector of problem orders
%% tau_vec = vector of tau parameter values in the SMM models
%% tol_vec = vector of ALM-SNCG tolerance
%% edgp = number of equally spaced grid points between the maximum and minimum values of C
%% datadir = path to the directory containing random data files
%==========================================================================================
function resultAvgWarm = Test_Warm_path_random(prob_vec,tau_vec,tol_vec,edgp,datadir)

C_vec = 10.^([-1:3/edgp:2]);
% ------------------------- input X and y ----------------------
filepath = fileparts(fileparts(datadir));
datadir2 =  [filepath,filesep,'Test2_path_Cvec'];
pathname = [datadir2,filesep,'Result_solution_path_figure_random',filesep,...
    'Result_Warm_path_random',filesep];
if ~exist(pathname,'dir'), mkdir(pathname); end
addpath(genpath(pathname));
lenqq = length(prob_vec);
lentau = length(tau_vec);
lenC = length(C_vec); rr = 14;
lentol = length(tol_vec);
resulthistWarm = zeros(lentau*lenqq*rr,lenC);
resultAvgWarm = zeros(lenqq*lentau,10);
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
    OPTIONS.ifrandom = 1;
    OPTIONS.flag_scaling = 0;
    OPTIONS.warm = 0;
    OPTIONS.sigmaiter = 2;
    OPTIONS.sigmascale = 1.2;
    OPTIONS.Test_sigma = 1;
    OPTIONS.tol_nSM = -1e-12;

    resultWarm_totletime_path = zeros(lentau,lenC);
    resultWarm_time_path = zeros(lentau,lenC);
    for oo = 1:lentol
        OPTIONS.tol = tol_vec(oo);
        for tt = 1:lentau
            OPTIONS.tau = tau_vec(tt);
            totletime = 0;
            for cc = 1:lenC
                OPTIONS.C = C_vec(cc);
                if n >= p*q
                    num_sig = 100;
                    max_sig = 3e3;
                    OPTIONS.sigma = min(num_sig*OPTIONS.C,max_sig);
                else
                    OPTIONS.sigma = 100;
                end

                if cc == 1
                    OPTIONS.W0 = zeros(Xy_train.p,Xy_train.q);
                    OPTIONS.b0 = 0;
                    OPTIONS.lam0 = zeros(OPTIONS.n,1);
                    OPTIONS.Lam0 = -OPTIONS.W0;
                    OPTIONS.v0 = ones(OPTIONS.n,1);
                    OPTIONS.U0 = OPTIONS.W0;
                else
                    OPTIONS.W0 = solution_W;
                    OPTIONS.b0 = solution_b;
                    OPTIONS.lam0 = solution_lam;
                    OPTIONS.Lam0 = solution_Lam;
                    OPTIONS.v0 = solution_v;
                    OPTIONS.U0 = solution_U;
                end

                fprintf('\n ============* qq = %2d, tt = %2d, cc = %2d *============',...
                    qqq, tt, cc);

                [obj,W,b,~,info] = ALMSNCG(Xy_train.X,Xy_train.y,OPTIONS);

                totletime = totletime + info.totaltime;
                resultWarm_totletime_path(tt+(qq-1)*lentau,cc) = totletime;
                resultWarm_time_path(tt+(qq-1)*lentau,cc) = info.totaltime;

                resulthistWarm(1+(tt-1)*rr+(qq-1)*lentau*rr,cc) = OPTIONS.n;
                resulthistWarm(2+(tt-1)*rr+(qq-1)*lentau*rr,cc) = OPTIONS.p;
                resulthistWarm(3+(tt-1)*rr+(qq-1)*lentau*rr,cc) = OPTIONS.q;
                resulthistWarm(4+(tt-1)*rr+(qq-1)*lentau*rr,cc) = OPTIONS.tau;
                resulthistWarm(5+(tt-1)*rr+(qq-1)*lentau*rr,cc) = OPTIONS.C;
                resulthistWarm(6+(tt-1)*rr+(qq-1)*lentau*rr,cc) = OPTIONS.sigma;
                resulthistWarm(7+(tt-1)*rr+(qq-1)*lentau*rr,cc) = info.nSMM;
                resulthistWarm(8+(tt-1)*rr+(qq-1)*lentau*rr,cc) = info.r_indexJ;
                resulthistWarm(9+(tt-1)*rr+(qq-1)*lentau*rr,cc) = info.res_kkt_final;
                resulthistWarm(10+(tt-1)*rr+(qq-1)*lentau*rr,cc) = info.res_gap_final;
                resulthistWarm(11+(tt-1)*rr+(qq-1)*lentau*rr,cc) = info.iter;
                resulthistWarm(12+(tt-1)*rr+(qq-1)*lentau*rr,cc) = info.numSSNCG;
                resulthistWarm(13+(tt-1)*rr+(qq-1)*lentau*rr,cc) = info.totaltime;
                resulthistWarm(14+(tt-1)*rr+(qq-1)*lentau*rr,cc) = obj(1);

                solution_W = W;
                solution_b = b;
                solution_lam = info.lam;
                solution_Lam = info.Lam;
                solution_v = info.v;
                solution_U = info.U;

            end
            maxrelkkt = max(resulthistWarm(9+(tt-1)*rr+(qq-1)*lentau*rr,2:end));
            resultAvgWarm_vec = mean(resulthistWarm(1+(tt-1)*rr+(qq-1)*lentau*rr:14+(tt-1)*rr+(qq-1)*lentau*rr,2:end),2);

            resultAvgWarm(tt+(oo-1)*lentau+(qq-1)*lentau*lentol,1) = OPTIONS.n;
            resultAvgWarm(tt+(oo-1)*lentau+(qq-1)*lentau*lentol,2) = OPTIONS.p;
            resultAvgWarm(tt+(oo-1)*lentau+(qq-1)*lentau*lentol,3) = OPTIONS.q;
            resultAvgWarm(tt+(oo-1)*lentau+(qq-1)*lentau*lentol,4) = OPTIONS.tol;
            resultAvgWarm(tt+(oo-1)*lentau+(qq-1)*lentau*lentol,5) = OPTIONS.tau;
            resultAvgWarm(tt+(oo-1)*lentau+(qq-1)*lentau*lentol,6) =  resultAvgWarm_vec(8); % Avg_|J_1|
            resultAvgWarm(tt+(oo-1)*lentau+(qq-1)*lentau*lentol,7) = maxrelkkt; % Worst_relkkt
            resultAvgWarm(tt+(oo-1)*lentau+(qq-1)*lentau*lentol,8) = resultAvgWarm_vec(11); % Avg_iter_alm
            resultAvgWarm(tt+(oo-1)*lentau+(qq-1)*lentau*lentol,9) = resultAvgWarm_vec(12); % Avg_ter_SNCG
            resultAvgWarm(tt+(oo-1)*lentau+(qq-1)*lentau*lentol,10) = resultAvgWarm_vec(13); % Avg_time
        end
    end
end


