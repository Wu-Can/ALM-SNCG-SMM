%%=============================================================================
%% Generate the figure of "Comparison of support matrices percentage
%% between reduced subproblems and original problems on the real data"
%% Input
%% prob_vec = 1, utilizing the CIFAR-10 dataset
%%          = 2, utilizing the MINIST dataset
%% tau_vec = vector of parameter tau values in the SMM model
%% tol_vec = vector of tolerence values epsilon for the SMM model
%% edgp = number of equally spaced grid points
%%=============================================================================
clc;clear
%%
%================================== INPUT =================================
prob_vec = [1 2];
tau_vec = [1 10];
tol_vec = [1e-4 1e-6];
edgp = 50;
%==========================================================================
lenp = length(prob_vec);
lentau = length(tau_vec);
lentol = length(tol_vec);
%%
for i = 1:lenp
    pp = prob_vec(i);
    if pp == 1
        n = 1e4; p = 32; q = 32;
        C_vec = 10.^([-3:3/edgp:0]);
        C_log_vec = log10(C_vec);
        Probname = 'CIFAR-10'; %'CIFAR-10 data';
    elseif pp == 2
        n = 6e4; p = 28; q = 28;
        C_vec = 10.^([-1:3/edgp:2]);
        C_log_vec = log10(C_vec);
        Probname = 'MNIST'; %'MNIST data';
    end
    lenC = length(C_vec);
    
    for oo = 1:lentol
        tol = tol_vec(oo);
        
       %% download "resultAS_nSMM_path" and "resultAS_nImean_path"
        eval(['load resultAS_nSMM_path_',num2str(n),'_',num2str(p),'_',num2str(q),'_',num2str(edgp),'_',num2str(tol),'.mat']);
        nSMM_path_AS = resultAS_nSMM_path;
        clear resultAS_nSMM_path;
        
        eval(['load resultAS_nImean_path_',num2str(n),'_',num2str(p),'_',num2str(q),'_',num2str(edgp),'_',num2str(tol),'.mat']);
        nImean_path_AS = resultAS_nImean_path;
        clear resultAS_nImean_path;
        
        figure(3)
        for tt = 1:lentau
            tau = tau_vec(tt);
            num_fig = tt + (oo-1)*lentau*lenp+(pp-1)*lentau;
            maxC = max(C_log_vec); minC = min(C_log_vec);
            lam_ed = 6;
            
            Fig_h(num_fig) = subplot(lentau,lentau*lenp,num_fig);
            nSM_vec = nSMM_path_AS(tt+(pp-1)*lentau,:);
            nImean_vec = nImean_path_AS(tt+(pp-1)*lentau,:);
            Matrix_percent = [nSM_vec./nImean_vec;nSM_vec/n].*100;
            %% draws the bars
            bar_figure = bar(C_log_vec([5:5:50]),Matrix_percent(:,[5:5:50]));
            bar_figure(1).FaceColor = [0.8500 0.3250 0.0980];
            bar_figure(2).FaceColor = [0.3010 0.7450 0.9330];
            
            xlabel('log_{10}(C)');
            ylabel('Percentage');
            ytickformat('percentage');
            xlim([minC,maxC]);
            epsilonStr = strrep(num2str(tol, '%.0e'), 'e-0', 'e-');
            title({['Percentage comparison on ',Probname];['(n, p, q, \tau, \epsilon) = (',...
                num2str(n),',',num2str(p),',',num2str(q),',',num2str(tau),',',epsilonStr,')']});
            set(gca,'XTick',[C_log_vec([5:5:50])],'FontSize',12)
        end
    end
    
end

Fig_per = legend(Fig_h(1), 'mean(|SM| / n_I)','mean(|SM|) / n');
set(Fig_per, 'FontSize', 12);
set(Fig_per, 'Units', 'normalized', 'Position', [0.515, 0.02, 0.0, 0.0], 'Orientation', 'horizontal');

saveas(Fig_per,'Percentage_comparison_real.png');

