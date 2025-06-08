%%=========================================================================
%% Test ALM-SNCG performance for the Support Matrix Machine (SMM) model 
%% with fixed C using real data
%%
%% result = Test_ALMSNCG_real(prob_vec,tau_vec,tol_vec,stop_flag,datadir)
%%
%% INPUT:
%% prob_vec = vector of problem orders
%% tau_vec = vector of tau parameter values in the SMM model
%% tol_vec = vector of ALM-SNCG tolerance
%% stop_flag = stopping criterion for ALM-SNCG:
%%             0: terminate when relkkt < tol (relative KKT residual)
%%             1: terminate when relobj < tol (relative objective change)
%%             2: terminate when relgap < tol (relative duality gap)
%% datadir = path to the directory containing real data files
%==========================================================================
function result = Test_ALMSNCG_real(prob_vec,tau_vec,tol_vec,stop_flag,datadir)

fname{1} = 'A_EEG_train';
fname{2} = 'A_train'; % INRIA
fname{3} = 'A_c5_c9_train'; % CIFAR10: dog or truck
fname{4} = 'A_train10_minist'; % MINIST: 0 or 1

if (stop_flag ~= 0) && (stop_flag ~= 1) && (stop_flag ~= 2)
    fprintf('The value of stop_flag must be 0, 1 or 2 !');
    return;
end
if stop_flag == 1
    datadir_opt = fileparts(datadir);
    addpath(genpath(datadir_opt));
    optobj = load([datadir_opt,filesep,'result_ALMSNCG_real_relkkt_1e-08.mat']);
end
%%
lenp = length(prob_vec);
lentau = length(tau_vec);
lentol = length(tol_vec);
result = zeros(lenp*lentol*lentau*4,11);

for pp = 1:lenp
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
                C_vec = [1e-3 1e-2 1e-1 1]; num_tmp = 4;
            case 3
                load([datadir,filesep,'A_c5_c9_test.mat']);
                C_vec = [1e-3 1e-2 1e-1 1]; num_tmp = 4;
            case 4
                load([datadir,filesep,'A_test10_minist']);
                C_vec = [1e-1 1 1e1 1e2]; num_tmp = 2;
        end
    else
        fprintf('\n Data file not found!');
        fprintf('\n ');
        return
    end
    lenC = length(C_vec);

    eval(['Ainput = ',fname{ii},'.Ainput;']);
    eval(['y = ',fname{ii},'.y;']);
    eval(['n = ',fname{ii},'.n;']);
    eval(['p = ',fname{ii},'.p;']);
    eval(['q = ',fname{ii},'.q;']);
    if ii >= 3
        Ainput = Ainput';
    end

    OPTIONS.flag_scaling = 1;
    OPTIONS.n = n;
    OPTIONS.p = p;
    OPTIONS.q = q;
    OPTIONS.sigmaiter = 2;
    OPTIONS.sigmascale = 1.2;
    OPTIONS.tol_admm = 1e-4;
    OPTIONS.ifrandom = 0;

    OPTIONS.stop = stop_flag;
    for oo = 1:lentol
        tol = tol_vec(oo);
        OPTIONS.tol = tol;

        for tt = 1:lentau
            OPTIONS.tau = tau_vec(tt);
            if OPTIONS.n < 1000
                if  OPTIONS.tau == 1
                    OPTIONS.sig0_admm = 50;
                else
                    OPTIONS.sig0_admm = 25;
                end
            else
                if  OPTIONS.tau == 1
                    OPTIONS.sig0_admm = 10;
                else
                    OPTIONS.sig0_admm = 15;
                end
            end

            for cc = 1:lenC
                OPTIONS.C = C_vec(cc);
                OPTIONS.warm = 0;
                OPTIONS.maxiter_admm = 0;

                if stop_flag
                    if ((n < 1e3) && (OPTIONS.C <= 1e-3)) || ((n >= 1e3)&&(OPTIONS.C <= 1e-2)&&(OPTIONS.tau == 10))
                        OPTIONS.warm = 1; OPTIONS.maxiter_admm = 10;
                    elseif ((n >= 1e3) && (p*q > 2*n) && (OPTIONS.tau == 10)) || ((n >= 1e3)&&(OPTIONS.C <= 1e-2)&&(OPTIONS.tau == 1))
                        OPTIONS.warm = 1; OPTIONS.maxiter_admm = 5;
                    end
                end

                if n < 1e3
                    if OPTIONS.C <= 1e-3
                        OPTIONS.sigma =  180;
                    else
                        OPTIONS.sigma =  25;
                    end
                elseif n <= 1e4
                    if OPTIONS.C <= 1e-2
                        OPTIONS.sigma =  0.05;
                    elseif n <= 5e3
                        OPTIONS.sigma = 0.01;
                    else
                        OPTIONS.sigma = 5;
                    end
                else
                    if OPTIONS.C <= 1
                        OPTIONS.sigma = 10;
                    else
                        OPTIONS.sigma = 200;% 15;
                    end
                end

                if OPTIONS.stop == 1
                    OPTIONS.optval = optobj.result(num_tmp+log10(OPTIONS.C)+log10(OPTIONS.tau)*4+(ii-1)*8,end);
                end


                [obj,W_train,b_train,~,info] = ALMSNCG(Ainput,y,OPTIONS);

                % Compute the accuracy on the test set
                switch ii
                    case 1
                        Y_test = A_EEG_test.Ainput'*W_train(:) + b_train;
                        y_test = mysign(Y_test);
                        accuracy_test = sum(y_test == A_EEG_test.y)/length(A_EEG_test.y);
                    case 2
                        Y_test = A_test.Ainput'*W_train(:) + b_train;
                        y_test = mysign(Y_test);
                        accuracy_test = sum(y_test == A_test.y)/length(A_test.y);
                    case 3
                        Y_test = A_c5_c9_test.Ainput*W_train(:) + b_train;
                        y_test = mysign(Y_test);
                        accuracy_test = sum(y_test == A_c5_c9_test.y)/length(A_c5_c9_test.y);
                    case 4
                        Y_test = A_test10_minist.Ainput*W_train(:) + b_train;
                        y_test = mysign(Y_test);
                        accuracy_test = sum(y_test == A_test10_minist.y)/length(A_test10_minist.y);
                end

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
end



