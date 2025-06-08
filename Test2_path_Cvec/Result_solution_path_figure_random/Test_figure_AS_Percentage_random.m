%%================================================================================
%% Generate the figure of "Comparison of support matrices percentage
%% between reduced subproblems and original problems on the random data"
%% Input
%% prob_vec = vector of problem orders    
%% tau_vec = vector of parameter tau values in the SMM model
%% edgp = number of equally spaced grid points
%% tol_vec = vector of tolerence values epsilon for the SMM model
%% edgp = number of equally spaced grid points
%% C_vec = the sequence C_i for i = 1,...,N
%%================================================================================
clc;clear
%%
%================================== INPUT =================================
prob_vec = [1:4];
tau_vec = [10 100];
edgp = 50;
tol_vec = [1e-4 1e-6];
C_vec = 10.^([-1:3/edgp:2]);
%==========================================================================
C_log_vec = log10(C_vec);
lentau = length(tau_vec);
lenp = length(prob_vec);
lentol = length(tol_vec);
lenC = length(C_vec);
%%
for i = 1:lenp
    pp = prob_vec(i);
    switch pp
        case 1
            n = 1e4; p = 1e2; q = 1e2;
        case 2
            n = 1e4; p = 1e3; q = 5e2;
        case 3
            n = 1e5; p = 50; q = 100;
        case 4
            n = 1e6; p = 50; q = 100;
    end

    for oo = 1:lentol
        tol = tol_vec(oo);
        
        %% download "resultAS_nSMM_path" and "resultAS_nImean_path"
        eval(['load resultAS_nSMM_random_path_',num2str(n),'_',num2str(p),'_',num2str(q),'_',num2str(edgp),'_',num2str(tol),'.mat']);
        nSMM_path_AS = resultAS_nSMM_path;
        clear resultAS_nSMM_path;
        
        eval(['load resultAS_nImean_random_path_',num2str(n),'_',num2str(p),'_',num2str(q),'_',num2str(edgp),'_',num2str(tol),'.mat']);
        nImean_path_AS = resultAS_nImean_path;
        clear resultAS_nImean_path;
        
        for tt = 1:lentau
            tau = tau_vec(tt);
            num_fig = pp + (tt-1)*lenp + (oo-1)*lenp*lentau; 
            
            maxC = max(C_log_vec); minC = min(C_log_vec);
            Fig_h(num_fig) = subplot(lentol*lentau,lenp,num_fig);
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
            title({['Percentage comparison on '];['(n, p, q, \tau,\epsilon) = (',...
                num2str(n),',',num2str(p),',',num2str(q),',',num2str(tau),',',epsilonStr,')']});
            set(gca,'XTick',[C_log_vec([5:5:50])],'FontSize',11)
        end
    end
    
end

Fig_per = legend(Fig_h(9), 'mean(|SM| / n_I)','mean(|SM|) / n');
set(Fig_per, 'FontSize', 12);
set(Fig_per, 'Units', 'normalized', 'Position', [0.5, 0.05, 0.0, 0.0], 'Orientation', 'horizontal');


saveas(Fig_per,'Percentage_comparison_random.png');


