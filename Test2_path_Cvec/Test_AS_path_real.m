%%==========================================================================================
%% Test AS+ALM performance for the Support Matrix Machine (SMM) models 
%% with a sequence {C_i}^N_{i=1} using real data
%%
%% resultAvgAS = Test_AS_path_real(prob_vec,tau_vec,tol_vec,edgp,datadir)
%%
%% INPUT:
%% prob_vec = vector of problem orders
%% tau_vec = vector of tau parameter values in the SMM models
%% tol_vec = vector of ALM-SNCG tolerance
%% edgp = number of equally spaced grid points between the maximum and minimum values of C
%% datadir = path to the directory containing real data files
%==========================================================================================
function resultAvgAS = Test_AS_path_real(prob_vec,tau_vec,tol_vec,edgp,datadir)

%% ------------------------- input X and y ----------------------
filepath = fileparts(fileparts(datadir));
datadir2 =  [filepath,filesep,'Test2_path_Cvec'];
pathname = [datadir2,filesep,'Result_solution_path_figure_real',...
    filesep,'Result_AS_path_real',filesep];
if ~exist(pathname,'dir'),mkdir(pathname); end
addpath(genpath(pathname));

fname{1} = 'A_c5_c9_train';
fname{2} = 'A_train10_minist';

lenqq = length(prob_vec);
lentau = length(tau_vec);
lentol = length(tol_vec);
lenC = edgp + 1; rr = 14;
resulthistAS = zeros(lenqq*lentau*rr,lenC);
result_num_AindexI = zeros(1,lentau*lenqq);
resultAvgAS = zeros(lenqq*lentau,rr);
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

    XXT = Ainput';
    clear Ainput;

    OPTIONS.n = n;
    OPTIONS.p = p;
    OPTIONS.q = q;
    OPTIONS.flag_scaling = 1;
    OPTIONS.sigmaiter = 2;
    OPTIONS.sigmascale = 1.2;
    OPTIONS.Test_sigma = 1;
    OPTIONS.C_vec = C_vec;
    OPTIONS.tol_nSM = -1e-12;
    OPTIONS.stop = 0;
    OPTIONS.ifrandom = 0;
    OPTIONS.warm = 0;

    lenC = length(OPTIONS.C_vec);
    resultAS_nSMM_path = zeros(lentau,lenC);
    resultAS_nImean_path = zeros(lentau,lenC);
    for oo = 1:lentol
        OPTIONS.tol = tol_vec(oo);
        for tt = 1:lentau
            OPTIONS.tau = tau_vec(tt);
            switch qqq
                case 1
                    if OPTIONS.tau == 1
                        OPTIONS.sigma = 3*ones(1,lenC);
                    else
                        OPTIONS.sigma = 15*ones(1,lenC);
                    end
                case 2
                    if OPTIONS.tau == 1
                        OPTIONS.sigma = 30*OPTIONS.C_vec;
                    else
                        OPTIONS.sigma = 40*OPTIONS.C_vec;
                    end
            end

            OPTIONS.W0 = zeros(p,q);
            OPTIONS.b0 = 0;
            OPTIONS.lam0 = zeros(OPTIONS.n,1);
            OPTIONS.Lam0 = -OPTIONS.W0;

            [objopt,~,~,runhist,info] = AS_ALM(XXT,y,OPTIONS);


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

            result_num_AindexI(tt+(qq-1)*lenqq) = info.num_AindexI;

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

        eval(['filename_nSMM = ''resultAS_nSMM_path_',num2str(n),'_',num2str(p),'_',num2str(q),'_',num2str(edgp),'_',num2str(OPTIONS.tol),''';']);
        eval(['filename_nImean = ''resultAS_nImean_path_',num2str(n),'_',num2str(p),'_',num2str(q),'_',num2str(edgp),'_',num2str(OPTIONS.tol),''';']);

        save([pathname,filename_nSMM,'.mat'],'resultAS_nSMM_path');
        save([pathname,filename_nImean,'.mat'],'resultAS_nImean_path');
    end
end


