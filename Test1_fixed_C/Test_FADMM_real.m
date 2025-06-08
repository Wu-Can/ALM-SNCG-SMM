%%=========================================================================
%% Test FADMM performance for the Support Matrix Machine (SMM) model 
%% with fixed C using real data
%%
%% result = Test_FADMM_real(prob_vec,tau_vec,tol_vec,datadir)
%%
%% INPUT:
%% prob_vec = vector of problem orders
%% tau_vec = vector of tau parameter values in the SMM model
%% tol_vec = vector of ALM-SNCG tolerance
%% datadir = path to the directory containing real data files
%==========================================================================
%clc; clear;
%profile on
function result = Test_FADMM_real(prob_vec,tau_vec,tol_vec,datadir)

fname{1} = 'A_EEG_train';
fname{2} = 'A_train'; % INRIA
fname{3} = 'A_c5_c9_train'; % CIFAR10: dog or truck
fname{4} = 'A_train10_minist'; % MINIST: 0 or 1

lenprob = length(prob_vec);
lentau = length(tau_vec);
lentol = length(tol_vec);
result = zeros(lenprob*lentol*lentau*4,9);
stop_flag = 1; % relative objective values as stopping criterion
if stop_flag == 1
    datadir_opt = fileparts(datadir);
    addpath(genpath(datadir_opt));
    optobj = load([datadir_opt,filesep,'result_ALMSNCG_real_relkkt_1e-08.mat']);
end
for pp = 1:lenprob
    ii = prob_vec(pp);
    probname = [datadir,filesep,fname{ii}];
    fprintf('\n Problem name: %s', fname{ii});
    if exist([probname,'.mat'])
        load([probname,'.mat'])
        switch ii
            case 1
                load([datadir,filesep,'A_EEG_test.mat']);
                C_vec = [1e-4 1e-3 1e-2 1e-1]; num_tmp = 5;
            case 2
                load([datadir,filesep,'A_test.mat']);
                C_vec = [1e-3 1e-2 1e-1 1]; 
                num_tmp = 4;
            case 3
                load([datadir,filesep,'A_c5_c9_test.mat']);
                C_vec = [1e-3 1e-2 1e-1 1]; num_tmp = 4;
            case 4
                load([datadir,filesep,'A_test10_minist']);
                C_vec = [1e-1 1 1e1 1e2]; num_tmp = 2;
        end
    else
        fprintf('\n Can not find the file');
        fprintf('\n ');
        return
    end
    lenC = length(C_vec);

    eval(['X = ',fname{ii},'.Ainput;']);
    eval(['y = ',fname{ii},'.y;']);
    eval(['n = ',fname{ii},'.n;']);
    eval(['p = ',fname{ii},'.p;']);
    eval(['q = ',fname{ii},'.q;']);

    if ii >= 3
        X = X';
    end
    max_iter = 30000;
    inner_iter = 10000;
    for oo = 1:lentol
        eps = tol_vec(oo);

        for tt = 1:lentau
            tau = tau_vec(tt);
            for cc = 1:lenC
                C = C_vec(cc);
                if stop_flag == 1
                     optval = optobj.result(num_tmp+log10(C)+log10(tau)*4+(ii-1)*8,end);
                end

                [W_train_vec, b_train, rk_W, iter, info] = fastADMM (X', y, p, q, C, tau, max_iter, inner_iter, eps, stop_flag, optval);

                % compute the accuracy on the training set
                Y_train = X'*W_train_vec + b_train;
                y_train = mysign(Y_train);
                right_y = (y_train == y);
                accuracy_train = sum(right_y)/n;

                % compute the number of support matrices
                xi = 1-y.*(Y_train);
                index_xigeq0 = (xi > -1e-12);
                nSMM = sum(index_xigeq0);

                % compute the accuracy on the test set
                switch ii
                    case 1
                        Y_test = A_EEG_test.Ainput'*W_train_vec + b_train;
                        y_test = mysign(Y_test);
                        accuracy_test = sum(y_test == A_EEG_test.y)/length(A_EEG_test.y);
                    case 2
                        Y_test = A_test.Ainput'*W_train_vec + b_train;
                        y_test = mysign(Y_test);
                        accuracy_test = sum(y_test == A_test.y)/length(A_test.y);
                    case 3
                        Y_test = A_c5_c9_test.Ainput*W_train_vec + b_train;
                        y_test = mysign(Y_test);
                        accuracy_test = sum(y_test == A_c5_c9_test.y)/length(A_c5_c9_test.y);
                    case 4
                        Y_test = A_test10_minist.Ainput*W_train_vec + b_train;
                        y_test = mysign(Y_test);
                        accuracy_test = sum(y_test == A_test10_minist.y)/length(A_test10_minist.y);
                end

                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,1) = n;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,2) = p;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,3) = q;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,4) = eps;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,5) = tau;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,6) = C;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,7) = accuracy_test;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,8) = info.relobj;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,9) = info.time;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,10) = iter;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,11) = info.time/iter;

            end
        end
    end
end