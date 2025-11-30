%%
%% Replicate the numerical results from Figure 3
%%
clc;clear;
HOME = pwd; addpath(genpath(HOME));
filepath = fileparts(HOME);
datadir =  [filepath,filesep,'Data\Real_data'];

maxiter_vec = [5e3:5e3:1e5]; % Range of maximum iteration numbers

%% Results of ALM-SNCG on EEG dataset
result_alm = Test_ALMSNCG_real_miter(1,1,1e-3,1e-3,1,maxiter_vec,datadir);

%% Results of SGD on EEG dataset
result_sgs = Test_SGD_real_miter(1, 1e-3, 1, 1e-3, maxiter_vec, 10, datadir);

%% Generate Figure 3
figure(1);
itemname={'$\mbox{Accuracy}_{test}$', 'Obj', 'Time', '$\mbox{Time}_{svd}$'};
for ii = 1:length(itemname)

    Fig_h(ii) = subplot(2,2,ii);
    switch ii
        case 1
            cc_sgs = 7; cc_alm = 9;
        case 2
            cc_sgs = 11; cc_alm = 18;
        case 3
            cc_sgs = 9; cc_alm = 11;
        case 4
            cc_sgs = 13; cc_alm = 17;
    end

    yy_sgs = result_sgs(:, cc_sgs);
    yy_alm = result_alm(:, cc_alm);

    xx = maxiter_vec/1e3;
    switch ii
        case 1
        h1_sgs = plot(xx, yy_sgs,'b-s','LineWidth',2,'MarkerSize',10);
        hold on
        h1_alm = plot(xx, yy_alm,'r-.*','LineWidth',2,'MarkerSize',10);
        hold off
        case 2
        h2_sgs = plot(xx, yy_sgs,'b-s','LineWidth',2,'MarkerSize',10);
        hold on
        h2_alm = plot(xx, yy_alm,'r-.*','LineWidth',2,'MarkerSize',10);
        hold off
        case 3
        h3_sgs = plot(xx, yy_sgs,'b-s','LineWidth',2,'MarkerSize',10);
        hold on
        h3_alm = plot(xx, yy_alm,'r-.*','LineWidth',2,'MarkerSize',10);
        hold off
        case 4
        h4_sgs = plot(xx, yy_sgs,'b-s','LineWidth',2,'MarkerSize',10);
        hold on
        h4_alm = plot(xx, yy_alm,'r-.*','LineWidth',2,'MarkerSize',10);
        hold off
    end

    if ii == 1
    h_plots = [h1_sgs, h1_alm];
    end

    title([itemname{ii},' on ', 'EEG with tau=1 and C=1e-3'],'Interpreter','latex','FontSize',24);
    xlabel('maxiter/$10^3$','Interpreter','latex','FontSize',20);
    ylabel(itemname{ii},'Interpreter','latex','FontSize',20);

end

Fig_per = legend(h_plots, {'SGD', 'ALM-SNCG'}, 'Interpreter', 'latex', 'FontSize', 20, ...
   'Orientation', 'horizontal', 'Location', 'southoutside');

set(Fig_per, 'Units', 'normalized', 'Position', [0.3, 0.01, 0.4, 0.05]);

%% Save Figure 3
set(gcf,'Units','normalized','Position',[0,0,1,1]);
exportgraphics(gcf,'Figure_ALM_SGD_EEG.png','Resolution',300);
