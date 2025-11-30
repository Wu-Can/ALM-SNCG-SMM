%%=======================================================================================
%% Test SGD performance for the Support Matrix Machine (SMM) model
%% with fixed C using real data
%%
%% result = Test_SGD_real_miter(prob_vec, tol, tau, C, maxiter_vec, datadir)
%%
%% INPUT:
%% prob_vec = vector of problem orders
%% tol = ALM-SNCG tolerance
%% tau = value of parameter tau in the SMM model
%% C = value of parameter C in the SMM model
%% maxiter_vec = range of maximum iteration numbers
%% repeat = number of repetitions
%% datadir = path to the directory containing real data files
%=======================================================================================
function result = Test_SGD_real_miter(prob_vec, tol, tau, C, maxiter_vec, repeat, datadir)

fname{1} = 'A_EEG_train';
fname{2} = 'A_train'; % INRIA
fname{3} = 'A_c5_c9_train'; % CIFAR10: dog or truck
fname{4} = 'A_train10_minist'; % MINIST: 0 or 1

datadir_opt = fileparts(datadir);
addpath(genpath(datadir_opt));
optobj = load([datadir_opt,filesep,'result_ALMSNCG_real_relkkt_1e-08.mat']);
%%
lenp = length(prob_vec);
lenmiter = length(maxiter_vec);
result = zeros(lenp*lenmiter,13);
result_sum = zeros(lenp*lenmiter,13);
for rr = 1:repeat
    for pp = 1:lenp
        ii = prob_vec(pp);
        probname = [datadir,filesep,fname{ii}];
        fprintf('\n Problem name: %s', fname{ii});
        if exist([probname,'.mat'])
            load([probname,'.mat'])
            switch ii
                case 1
                    load([datadir,filesep,'A_EEG_test.mat']); num_tmp = 5;
                case 2
                    load([datadir,filesep,'A_test.mat']); num_tmp = 4;
                case 3
                    load([datadir,filesep,'A_c5_c9_test.mat']); num_tmp = 4;
                case 4
                    load([datadir,filesep,'A_test10_minist']);num_tmp = 2;
            end
        else
            fprintf('\n Data file not found!');
            fprintf('\n ');
            return
        end

        eval(['Ainput = ',fname{ii},'.Ainput;']);
        eval(['y = ',fname{ii},'.y;']);
        eval(['n = ',fname{ii},'.n;']);
        eval(['p = ',fname{ii},'.p;']);
        eval(['q = ',fname{ii},'.q;']);
        if ii >= 3
            Ainput = Ainput';
        end

        OPTIONS.n = n;
        OPTIONS.p = p;
        OPTIONS.q = q;
        OPTIONS.test_svd = 1; % time spent on SVD
        OPTIONS.flag_proj = 0; %not use projection step
        OPTIONS.tol = tol;

        for oo = 1:lenmiter
            OPTIONS.maxiter = maxiter_vec(oo);
            

            OPTIONS.tau = tau(pp);
            OPTIONS.C = C(pp);
            OPTIONS.optval = optobj.result(num_tmp+log10(OPTIONS.C)+log10(OPTIONS.tau)*4+(ii-1)*8,end);


            [obj, W_train, b_train, ~, info] = SGD(Ainput,y,OPTIONS);

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

            result(oo+(pp-1)*lenmiter,1) = OPTIONS.n;
            result(oo+(pp-1)*lenmiter,2) = OPTIONS.p;
            result(oo+(pp-1)*lenmiter,3) = OPTIONS.q;
            result(oo+(pp-1)*lenmiter,4) = OPTIONS.tol;
            result(oo+(pp-1)*lenmiter,5) = OPTIONS.tau;
            result(oo+(pp-1)*lenmiter,6) = OPTIONS.C;
            result(oo+(pp-1)*lenmiter,7) = accuracy_test;
            result(oo+(pp-1)*lenmiter,8) = info.relobj;
            result(oo+(pp-1)*lenmiter,9) = info.totaltime;
            result(oo+(pp-1)*lenmiter,10) = info.iter;
            result(oo+(pp-1)*lenmiter,11) = obj;
            result(oo+(pp-1)*lenmiter,12) = info.num_svd;
            result(oo+(pp-1)*lenmiter,13) = info.time_svd;
        end
    end
    result_sum = result_sum + result;
end

result = (1/repeat)*result_sum;

%save result_SGD_miter.mat result



