%%=========================================================================
%% Test FADMM performance for the Support Matrix Machine (SMM) model 
%% with fixed C using random data
%%
%% result = Test_FADMM_random(prob_vec,tau_vec,tol_vec,datadir)
%%
%% INPUT:
%% prob_vec = vector of problem orders
%% tau_vec = vector of tau parameter values in the SMM model
%% tol_vec = vector of ALM-SNCG tolerance
%% datadir = path to the directory containing random data files
%==========================================================================
function result = Test_FADMM_random(prob_vec,tau_vec,tol_vec,datadir)

C_vec = [0.1 1 10 100];

datadir_opt = fileparts(datadir);
addpath(genpath(datadir_opt));
prob_optobj = [datadir_opt,filesep,'result_ALMSNCG_random_relkkt_1e-08.mat'];
optobj = load(prob_optobj);

lenpp = length(prob_vec);
lentau = length(tau_vec);
lenC = length(C_vec);
lentol = length(tol_vec);
result = zeros(lenpp*lentau*lenC,9);
stop_flag = 1;
%%

for pp = 1:lenpp
    ppp = prob_vec(pp);
    switch ppp
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

    max_iter = 30000;
    inner_iter = 10000;
    for oo = 1:lentol
        tol = tol_vec(oo);
        eps = tol;

        for tt = 1:lentau
            tau = tau_vec(tt);
            for cc = 1:lenC
                C = C_vec(cc);
                if stop_flag
                    optval = optobj.result(2+log10(C)+(log10(tau)-1)*4+(ppp-1)*8,end);
                end

                [W_train_vec, b_train, rk_W, iter, info] = fastADMM((Xy_train.X)', Xy_train.y, p, q, C, tau, max_iter, inner_iter, eps, stop_flag, optval);

                % compute the classification accuracy on the training set
                Y_train = (W_train_vec'*Xy_train.X)' + b_train;
                y_train_com = mysign(Y_train);
                right_y = (y_train_com == Xy_train.y);
                accuracy_train = sum(right_y)/n;

                % compute the number of support matrices
                xi = 1-Xy_train.y.*(Y_train);
                index_xigeq0 = (xi >= -1e-12);
                nSMM = sum(index_xigeq0);

                % compute the classification accuracy on the test set
                Y_test = (W_train_vec'*Xy_test.X)' + b_train;
                y_test_com = mysign(Y_test);
                accuracy_test = sum(y_test_com ==Xy_test.y)/Xy_test.n;

                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,1) = n;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,2) = p;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,3) = q;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,4) = tol;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,5) = tau;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,6) = C;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,7) = accuracy_test;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,8) = info.relobj;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,9) = info.time;

            end
        end
    end
    clear Xy_train Xy_test
end

