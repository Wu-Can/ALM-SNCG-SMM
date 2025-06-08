%%=========================================================================
%% Test ALM-SNCG performance for the Support Matrix Machine (SMM) model 
%% with fixed C using random data
%%
%% result = Test_ALMSNCG_random(prob_vec,tau_vec,tol_vec,stop_flag,datadir)
%%
%% INPUT:
%% prob_vec = vector of problem orders
%% tau_vec = vector of tau parameter values in the SMM model
%% tol_vec = vector of ALM-SNCG tolerance
%% stop_flag = stopping criterion for ALM-SNCG:
%%             0: terminate when relkkt < tol (relative KKT residual)
%%             1: terminate when relobj < tol (relative objective change)
%%             2: terminate when relgap < tol (relative duality gap)
%% datadir = path to the directory containing random data files
%==========================================================================
function result = Test_ALMSNCG_random(prob_vec,tau_vec,tol_vec,stop_flag,datadir)
C_vec = [0.1 1 10 100];
%% Input the objective values under relkkt < 1e-8 if stop_flag = 1
if stop_flag == 1
    datadir_opt = fileparts(datadir);
    addpath(genpath(datadir_opt));
    prob_optobj = [datadir_opt,filesep,'result_ALMSNCG_random_relkkt_1e-08.mat'];
    optobj = load(prob_optobj);
end

lenProb = length(prob_vec);
lentau = length(tau_vec);
lenC = length(C_vec);
lentol = length(tol_vec);
result = zeros(lenProb*lentol*lentau*lenC,11);


for pp = 1:lenProb
    prob = prob_vec(pp);
    switch prob
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


    OPTIONS.tol_admm = 1e-4;
    OPTIONS.ifrandom = 1;
    OPTIONS.stop = stop_flag;
    OPTIONS.n = Xy_train.n;
    OPTIONS.p = Xy_train.p;
    OPTIONS.q = Xy_train.q;
    OPTIONS.ifrandom = 1;
    OPTIONS.flag_scaling = 0;
    OPTIONS.sigmaiter = 2;
    OPTIONS.sigmascale = 1.2;
    OPTIONS.Test_sigma = 1;

    for oo = 1:lentol
        tol = tol_vec(oo);
        OPTIONS.tol = tol;

        for tt = 1:lentau
            OPTIONS.tau = tau_vec(tt);
            for cc = 1:lenC
                OPTIONS.C = C_vec(cc);

                if OPTIONS.n >= OPTIONS.p*OPTIONS.q
                    OPTIONS.warm = 0;
                    if OPTIONS.C <= 1
                        OPTIONS.sigma = 10;
                    elseif OPTIONS.C <= 10
                        OPTIONS.sigma = 100;
                    else
                        OPTIONS.sigma = 900;
                    end
                else
                    OPTIONS.warm = 1;
                    OPTIONS.maxiter_admm = 1;
                    OPTIONS.sig0_admm = 0.1;
                    if OPTIONS.C >= 100
                        OPTIONS.maxiter_admm = 4;
                        OPTIONS.sig0_admm = 4;
                    end
                    if OPTIONS.tau <= 10
                        if OPTIONS.C <= 1
                            OPTIONS.sigma = 1;
                        else
                            OPTIONS.sigma = 50;
                        end
                    else
                        OPTIONS.sigma = 100;
                    end
                end
                if stop_flag ~= 1
                    OPTIONS.warm = 0;
                end

                if OPTIONS.C <= 1
                    OPTIONS.sigalm_scale1 = 1;
                    OPTIONS.sigalm_scale2 = 2;
                    OPTIONS.sigalm_scale3 = 10;
                else
                    OPTIONS.sigalm_scale1 = 10;
                    OPTIONS.sigalm_scale2 = 25;
                    OPTIONS.sigalm_scale3 = 50;
                end

                if OPTIONS.stop
                    OPTIONS.optval = optobj.result(2+log10(OPTIONS.C)+(log10(OPTIONS.tau)-1)*4+(prob-1)*8,end);
                end

                [obj,W_train,b_train,~,info] = ALMSNCG(Xy_train.X,Xy_train.y,OPTIONS);

                % compute the accuracy on the test set
                Y_test = (W_train(:)'*Xy_test.X)' + b_train;
                y_test_com = mysign(Y_test);
                accuracy_test = sum(y_test_com == Xy_test.y)/Xy_test.n;

                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,1) = OPTIONS.n;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,2) = OPTIONS.p;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,3) = OPTIONS.q;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,4) = OPTIONS.tol;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,5) = OPTIONS.tau;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,6) = OPTIONS.C;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,7) = info.r_indexJ;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,8) = info.k1;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,9) = accuracy_test;
                if OPTIONS.stop == 1
                    result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,10) = info.relobj;
                end
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,11) = info.totaltime;
                if tol == 1e-8
                    result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,12) = obj(1);
                end
            end
        end
        
    end
    clear Xy_train Xy_test

end





