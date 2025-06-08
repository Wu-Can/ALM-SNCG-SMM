%%=========================================================================
%% Test sGS-isPADMM performance for the Support Matrix Machine (SMM) model 
%% with fixed C using real data
%%
%% result = Test_sGS_isPADMM_real(prob_vec,tau_vec,tol_vec,datadir)
%%
%% INPUT:
%% prob_vec = vector of problem orders
%% tau_vec = vector of tau parameter values in the SMM model
%% tol_vec = vector of ALM-SNCG tolerance
%% datadir = path to the directory containing random data files
%==========================================================================
function result = Test_sGS_isPADMM_real(prob_vec,tau_vec,tol_vec,datadir)

fname{1} = 'A_EEG_train';
fname{2} = 'A_train'; % INRIA
fname{3} = 'A_c5_c9_train'; % CIFAR10: dog or truck
fname{4} = 'A_train10_minist'; % MINIST: 0 or 1

len_prob = length(prob_vec);
lentau = length(tau_vec);
lentol = length(tol_vec);
result = zeros(len_prob*lentau*4,19);
datadir_opt = fileparts(datadir);
addpath(genpath(datadir_opt));
optobj = load([datadir_opt,filesep,'result_ALMSNCG_real_relkkt_1e-08.mat']);
%%
for pp = 1:len_prob
    ii = prob_vec(pp);

    probname = [datadir,filesep,fname{ii}];
    fprintf('\n Problem name: %s', fname{ii});
    if exist([probname,'.mat'])
        load([probname,'.mat'])
        switch ii
            case 1
                load([datadir,filesep,'A_EEG_test.mat']);
                C_vec = [1e-4 1e-3 1e-2 1e-1];
                num_tmp = 5;
            case 2
                load([datadir,filesep,'A_test.mat']);
                C_vec = [1e-3 1e-2 1e-1 1];num_tmp = 4;
            case 3
                load([datadir,filesep,'A_c5_c9_test.mat']);
                C_vec = [1e-3 1e-2 1e-1 1];num_tmp = 4;
            case 4
                load([datadir,filesep,'A_test10_minist']);
                C_vec = [1e-1 1 1e1 1e2]; num_tmp = 2;
        end
    else
        fprintf('\n Can not find the file!');
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
    %--------------------------
    OPTIONS.stop = 1; %  relative KKT if stop=0; Otherwise, relative objective values
    OPTIONS.computeW = 1;
    OPTIONS.largenum_k = 25;
    OPTIONS.flag_scaling = 1;
    OPTIONS.n = n;
    OPTIONS.p = p;
    OPTIONS.q = q;
    OPTIONS.fixsigma = 0;
    OPTIONS.ifrandom = 0;

    OPTIONS.steplen = 1.618;
    OPTIONS.sigma_siter = 50;
    OPTIONS.sigma_giter = 500;
    %OPTIONS.sigma = 1e-1;%1;

    OPTIONS.sigma_iter = 100;
    OPTIONS.sigma_mul = 20;
    OPTIONS.sigscale1 = 1.2;

    OPTIONS.sigscale2 = 3.5;
    OPTIONS.sigscale3 = 4.0;

    for oo = 1:lentol
        OPTIONS.tol = tol_vec(oo);
        for tt = 1:lentau
            OPTIONS.tau = tau_vec(tt);
            for cc = 1:lenC
                OPTIONS.C = C_vec(cc);
                if n < 500
                    OPTIONS.sigma = min(max(1,1e-2/OPTIONS.C),10);
                elseif n < 1e4
                    OPTIONS.sigma = min(max(1e-2, OPTIONS.C),0.1);
                elseif n < 5e4
                    OPTIONS.sigma = (1+log10(OPTIONS.tau))*5e-2;
                else
                    OPTIONS.sigma = min(max(1e-2, 1e-2*OPTIONS.C),0.1);
                end
                OPTIONS.optval = optobj.result(num_tmp+log10(OPTIONS.C)+log10(OPTIONS.tau)*4+(ii-1)*8,end);

                [~,~,~,~,info] = sGS_isPADMM(Ainput,y,OPTIONS);

                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,1) = OPTIONS.n;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,2) = OPTIONS.p;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,3) = OPTIONS.q;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,4) = OPTIONS.tol;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,5) = OPTIONS.tau;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,6) = OPTIONS.C;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,7) = info.relobj;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,8) = info.iter;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,9) = info.totaltime;
                result(cc+(tt-1)*lenC+(oo-1)*lenC*lentau+(pp-1)*lenC*lentau*lentol,10) = info.totaltime/info.iter;

            end
        end
    end
    
end

