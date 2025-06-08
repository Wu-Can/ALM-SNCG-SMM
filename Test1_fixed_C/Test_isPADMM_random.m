%%=========================================================================
%% Test isPADMM performance for the Support Matrix Machine (SMM) model 
%% with fixed C using random data
%%
%% result = Test_isPADMM_random(prob_vec,tau_vec,tol_vec,datadir)
%%
%% INPUT:
%% prob_vec = vector of problem orders
%% tau_vec = vector of tau parameter values in the SMM model
%% tol_vec = vector of ALM-SNCG tolerance
%% datadir = path to the directory containing random data files
%==========================================================================
function result = Test_isPADMM_random(prob_vec,tau_vec,tol_vec,datadir)

C_vec = [0.1 1 10 100];

datadir_opt = fileparts(datadir);
addpath(genpath(datadir_opt));
prob_optobj = [datadir_opt,filesep,'result_ALMSNCG_random_relkkt_1e-08.mat'];
optobj = load(prob_optobj);
lenProb = length(prob_vec);
lentau = length(tau_vec);
lenC = length(C_vec);
lentol = length(tol_vec);
result = zeros(lenProb*lentau*lenC,9);
%%

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

    OPTIONS.n = Xy_train.n;
    OPTIONS.p = Xy_train.p;
    OPTIONS.q = Xy_train.q;
    OPTIONS.fixsigma = 0;
    OPTIONS.flag_scaling = 0;
    OPTIONS.ifrandom = 1;
    OPTIONS.steplen = 1.618;
    OPTIONS.sigma_siter = 50;
    OPTIONS.sigma_giter = 500;
    OPTIONS.sigma_iter = 5;
    OPTIONS.sigma_mul = 20;
    OPTIONS.sigscale1 = 1.4;
    OPTIONS.sigscale2 = 1.05;
    OPTIONS.sigscale3 = 1.25;
    OPTIONS.sigma = 4;
    for oo = 1:lentol
        tol = tol_vec(oo);
        OPTIONS.tol = tol;

        for tt = 1:lentau
            OPTIONS.tau = tau_vec(tt);

            for cc = 1:lenC
                OPTIONS.C = C_vec(cc);

                if OPTIONS.C <= 1
                    OPTIONS.sigalm_scale1 = 1;
                    OPTIONS.sigalm_scale2 = 2;
                    OPTIONS.sigalm_scale3 = 10;
                else
                    OPTIONS.sigalm_scale1 = 10;
                    OPTIONS.sigalm_scale2 = 25;
                    OPTIONS.sigalm_scale3 = 50;
                end

                OPTIONS.optval = optobj.result(2+log10(OPTIONS.C)+(log10(OPTIONS.tau)-1)*4+(prob-1)*8,end);


                [obj,W_train,b_train,runhist,info] = isPADMM(Xy_train.X,Xy_train.y,OPTIONS);

                % compute the accuracy on the training set
                Y_train = (W_train(:)'*Xy_train.X)' + b_train;
                y_train_com = mysign(Y_train);
                right_y = (y_train_com == Xy_train.y);
                accuracy_train = sum(right_y)/OPTIONS.n;

                % compute the number of support matrices
                xi = 1-Xy_train.y.*(Y_train);
                index_xigeq0 = (xi >= -1e-12);
                numSMM = sum(index_xigeq0);

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
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,7) = accuracy_test;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,8) = info.relobj;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,9) = info.totaltime;

            end
        end
    end
    eval(['save result_isPADMM_random_relobj','_',num2str(tol),'.mat result']);
end




