%%=============================================================================
%% Generate Figure 2: The changes of values of \bar{k} and Accuracy_test as the 
%% value of \tau increases on EEG and CIFAR-10 datasets "
%% Input
%% C_vec = vector of parameter C values in the SMM model
%% tau_vec = vector of parameter tau values in the SMM model
%%=============================================================================

clc; clear;
HOME = pwd; addpath(genpath(HOME));
filepath = fileparts(HOME);
%================================== INPUT =================================
C_vec = [1e-3 1e-2 1e-1];
tau_vec = 10.^([-2:0.2:1]);
%==========================================================================

%% Input the training and test sets for the EEG and CIFAR-10 datasets
fname{1} = 'A_EEG_train';
fname{2} = 'A_c5_c9_train'; 
fname{3} = 'A_EEG_test';
fname{4} = 'A_c5_c9_test';

lenpro = length(fname)/2;
lenC = length(C_vec);
lentau = length(tau_vec);
k_bar_matrix = zeros(lenpro*lenC,lentau);
result = zeros(lenpro*lenC*lentau,21);
accuracy_test_matrix = zeros(lenpro*lenC,lentau);

%% Generate the k_bar_matrix and accuracy_test_matrix
for ii = 1:lenpro
    datadir_train = [filepath,filesep,'Data',filesep,'Real_data',filesep,fname{ii},'.mat'];
    load(datadir_train)
    datadir_test = [filepath,filesep,'Data',filesep,'Real_data',filesep,fname{ii+2},'.mat'];
    load(datadir_test)

    eval(['Ainput = ',fname{ii},'.Ainput;']);
    eval(['y = ',fname{ii},'.y;']);
    eval(['n = ',fname{ii},'.n;']);
    eval(['p = ',fname{ii},'.p;']);
    eval(['q = ',fname{ii},'.q;']);

    if ii >= 2
        Ainput = Ainput';
    end

    for cc = 1:lenC
        OPTIONS.C = C_vec(cc);

        for tt = 1:lentau
            OPTIONS.tau = tau_vec(tt);

            OPTIONS.tol = 1e-8;
            OPTIONS.flag_scaling = 1;
            OPTIONS.n = n;
            OPTIONS.p = p;
            OPTIONS.q = q;
            OPTIONS.sigmaiter = 2;
            OPTIONS.sigmascale = 1.2;
            OPTIONS.ifrandom = 0;
            OPTIONS.stop = 0;
            OPTIONS.warm = 0;
            OPTIONS.sigma =  0.1;

            [obj,W_train,b_train,runhist,info] = ALMSNCG(Ainput,y,OPTIONS);

            % Compute the accuracy on the training set
            Y_train = Ainput'*W_train(:) + b_train;
            y_train = mysign(Y_train);
            right_y = (y_train == y);
            accuracy_train = sum(right_y)/OPTIONS.n;
            %accuracy_train_matrix(cc+(ii-1)*lenC,tt) = accuracy_train;

            % Compute the number of support matrices
            xi = 1-y.*(Y_train);
            index_xigeq0 = (xi >= -1e-12);%(xi >= OPTIONS.tol);
            nSMM1 = sum(index_xigeq0);

            % Compute the accuracy on the test set
            switch ii
                case 1
                    Y_test = A_EEG_test.Ainput'*W_train(:) + b_train;
                    y_test = mysign(Y_test);
                    accuracy_test = sum(y_test == A_EEG_test.y)/length(A_EEG_test.y);
                case 2
                    Y_test = A_c5_c9_test.Ainput*W_train(:) + b_train;
                    y_test = mysign(Y_test);
                    accuracy_test = sum(y_test == A_c5_c9_test.y)/length(A_c5_c9_test.y);
            end
            accuracy_test_matrix(cc+(ii-1)*lenC,tt) = accuracy_test;


            % Compute k_bar
            negATlam = - ATzfun(Ainput,y,info.lam,OPTIONS.p,OPTIONS.q);
            [U,S,V] = svd(negATlam,'econ');
            nu = diag(S);
            index_nu = (nu > OPTIONS.tau + 1e-14);
            k_bar_matrix(cc+(ii-1)*lenC,tt) = sum(index_nu);

            result(tt+(cc-1)*lentau+(ii-1)*lentau*lenC,1) = n;
            result(tt+(cc-1)*lentau+(ii-1)*lentau*lenC,2) = p;
            result(tt+(cc-1)*lentau+(ii-1)*lentau*lenC,3) = q;
            result(tt+(cc-1)*lentau+(ii-1)*lentau*lenC,4) = OPTIONS.C;
            result(tt+(cc-1)*lentau+(ii-1)*lentau*lenC,5) = OPTIONS.tau;
            result(tt+(cc-1)*lentau+(ii-1)*lentau*lenC,6) = OPTIONS.sigma;
            result(tt+(cc-1)*lentau+(ii-1)*lentau*lenC,7) = info.r_indexJ;
            result(tt+(cc-1)*lentau+(ii-1)*lentau*lenC,8) = rank(W_train);
            result(tt+(cc-1)*lentau+(ii-1)*lentau*lenC,9) = info.k1;
            result(tt+(cc-1)*lentau+(ii-1)*lentau*lenC,10) = sum(index_nu);
            result(tt+(cc-1)*lentau+(ii-1)*lentau*lenC,11) = accuracy_train;
            result(tt+(cc-1)*lentau+(ii-1)*lentau*lenC,12) = accuracy_test;
            result(tt+(cc-1)*lentau+(ii-1)*lentau*lenC,13) = info.res_gap_final;
            result(tt+(cc-1)*lentau+(ii-1)*lentau*lenC,14) = info.res_kkt_final;
            result(tt+(cc-1)*lentau+(ii-1)*lentau*lenC,15) = info.iter;
            result(tt+(cc-1)*lentau+(ii-1)*lentau*lenC,16) = info.numSSNCG;
            result(tt+(cc-1)*lentau+(ii-1)*lentau*lenC,17) = info.numCG;
            result(tt+(cc-1)*lentau+(ii-1)*lentau*lenC,18) = info.num_smallalp;
            result(tt+(cc-1)*lentau+(ii-1)*lentau*lenC,19) = nSMM1;
            result(tt+(cc-1)*lentau+(ii-1)*lentau*lenC,20) = info.totaltime;
            result(tt+(cc-1)*lentau+(ii-1)*lentau*lenC,21) = obj(1);

        end
    end
end

%% Plot the figures of \bar{k} and Accuracy_test for the EEG and CIFAR-10 datasets
xx = log10(tau_vec);
dataname = {'EEG','CIFAR-10'};
for ii = 1:lenpro

    subplot(2,lenpro,ii);

    k_bar1 = k_bar_matrix(1+(ii-1)*lenC,:);
    k_bar2 = k_bar_matrix(2+(ii-1)*lenC,:);
    k_bar3 = k_bar_matrix(3+(ii-1)*lenC,:);

    plot(xx,k_bar1,'k:s','LineWidth',2,'MarkerSize',10);
    hold on;
    plot(xx,k_bar2,'r-d','LineWidth',2,'MarkerSize',10);
    hold on;
    plot(xx,k_bar3,'b--o','LineWidth',2,'MarkerSize',10);
    hold off;

    title(['$\bar{k}$ on ',dataname{ii}],'Interpreter','latex','FontSize',24);
    xlabel('$\mbox{log}_{10}(\tau)$','Interpreter','latex','FontSize',20);
    ylabel('$\bar{k}$','Interpreter','latex','FontSize',20);
    legend({'$C=10^{-3}$','$C=10^{-2}$','$C=10^{-1}$'},'Interpreter','latex','Location','northeast','FontSize',20);
    k_bar_max = round(0.1*max([k_bar1,k_bar2,k_bar3]))*10;
    ylim([0, k_bar_max]);
    yticks([0:k_bar_max/5:k_bar_max]);

    subplot(2,lenpro,ii+2)

    accu_test1 = accuracy_test_matrix(1+(ii-1)*lenC,:);
    accu_test2 = accuracy_test_matrix(2+(ii-1)*lenC,:);
    accu_test3 = accuracy_test_matrix(3+(ii-1)*lenC,:);

    plot(xx,accu_test1,'k:s','LineWidth',2,'MarkerSize',10);
    hold on;
    plot(xx,accu_test2,'r-d','LineWidth',2,'MarkerSize',10);
    hold on;
    plot(xx,accu_test3,'b--o','LineWidth',2,'MarkerSize',10);
    hold off;

    title(['${\mbox{Accuracy}}_{\mbox{test}}$ on ',dataname{ii}],'Interpreter','latex','FontSize',24);
    xlabel('$\mbox{log}_{10}(\tau)$','Interpreter','latex','FontSize',20);
    ylabel('${\mbox{Accuracy}}_{\mbox{test}}$','Interpreter','latex','FontSize',20);
    legend({'$C=10^{-3}$','$C=10^{-2}$','$C=10^{-1}$'},'Interpreter','latex','Location','southwest','FontSize',20);
    accu_test_max = ceil(100*max([accu_test1,accu_test2,accu_test3]))/100;
    accu_test_min = floor(100*min([accu_test1,accu_test2,accu_test3]))/100;
    ylim([accu_test_min, accu_test_max]);
    yticks([accu_test_min:(accu_test_max-accu_test_min)/5:accu_test_max]);

end

%% Save Figure 2
set(gcf,'Units','normalized','Position',[0,0,1,1]);
exportgraphics(gcf,'Figure_rank_W.png','Resolution',300);







