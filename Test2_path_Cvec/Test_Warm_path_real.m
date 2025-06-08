%%==========================================================================================
%% Test Warm+ALM performance for the Support Matrix Machine (SMM) models 
%% with a sequence {C_i}^N_{i=1} using real data
%%
%% resultAvgWarm = Test_Warm_path_real(prob_vec,tau_vec,tol_vec,edgp,datadir)
%%
%% INPUT:
%% prob_vec = vector of problem orders
%% tau_vec = vector of tau parameter values in the SMM models
%% tol_vec = vector of ALM-SNCG tolerance
%% edgp = number of equally spaced grid points between the maximum and minimum values of C
%% datadir = path to the directory containing real data files
%==========================================================================================
function resultAvgWarm = Test_Warm_path_real(prob_vec,tau_vec,tol_vec,edgp,datadir)

%% ------------------------- input X and y ----------------------
filepath = fileparts(fileparts(datadir));
datadir2 =  [filepath,filesep,'Test2_path_Cvec'];
pathname = [datadir2,filesep,'Result_solution_path_figure_real',...
    filesep,'Result_Warm_path_real',filesep];
if ~exist(pathname,'dir'),mkdir(pathname); end
addpath(genpath(pathname));

fname{1} = 'A_c5_c9_train';
fname{2} = 'A_train10_minist';

lenqq = length(prob_vec);
lentau = length(tau_vec);
lentol = length(tol_vec);
lenC = edgp + 1; rr = 14;
resulthistWarm = zeros(lentau*lenqq*rr,lenC);
resultAvgWarm = zeros(lenqq*lentau,10);
%%
for qq = 1:lenqq
    qqq = prob_vec(qq);
    switch qqq
        case 1
            C_vec = 10.^([-3:3/edgp:0]);
        case 2
            C_vec = 10.^([-1:3/edgp:2]);
    end

    probname = [datadir,filesep,fname{qqq}];
    fprintf('\n Problem name: %s', fname{qqq});
    if exist([probname,'.mat'])
        load([probname,'.mat'])
    else
        fprintf('\n Can not find the file');
        fprintf('\n ');
        return
    end
    eval(['Ainput = ',fname{qqq},'.Ainput;']);
    eval(['y = ',fname{qqq},'.y;']);
    eval(['n = ',fname{qqq},'.n;']);
    eval(['p = ',fname{qqq},'.p;']);
    eval(['q = ',fname{qqq},'.q;']);

    OPTIONS.n = n;
    OPTIONS.p = p;
    OPTIONS.q = q;
    OPTIONS.ifrandom = 0;
    OPTIONS.flag_scaling = 0;
    OPTIONS.warm = 0;
    OPTIONS.sigmaiter = 2;
    OPTIONS.sigmascale = 1.2;
    OPTIONS.Test_sigma = 1;
    OPTIONS.tol_nSM = -5e-1;
    XX = Ainput';
    clear Ainput;

    resultWarm_totletime_path = zeros(lentau,lenC);
    resultWarm_time_path = zeros(lentau,lenC);
    for oo = 1:lentol
        OPTIONS.tol = tol_vec(oo);
        for tt = 1:lentau
            OPTIONS.tau = tau_vec(tt);
            totletime = 0;
            for cc = 1:lenC
                OPTIONS.C = C_vec(cc);
                if n <= 1e4
                    if OPTIONS.tau == 1
                        OPTIONS.sigma = 3;
                    else
                        OPTIONS.sigma = 15;
                    end
                else
                    if OPTIONS.tau == 1
                        OPTIONS.sigma = 30*OPTIONS.C;
                    else
                        OPTIONS.sigma = 40*OPTIONS.C;
                    end
                end

                if cc == 1
                    OPTIONS.W0 = zeros(p,q);
                    OPTIONS.b0 = 0;
                    OPTIONS.lam0 = zeros(OPTIONS.n,1);
                    OPTIONS.Lam0 = -OPTIONS.W0;
                    OPTIONS.v0 = ones(OPTIONS.n,1);
                    OPTIONS.U0 = zeros(p,q);
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

                [obj,W,b,~,info] = ALMSNCG(XX,y,OPTIONS);

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



