%%=========================================================================
%% Generate Figure 1: Comparison of SM, ASM, and NSM
%% Input
%% tau = parameter tau in the SMM model
%% C_vec = vector of parameter C values in the SMM model
%%=========================================================================

clc;clear;
%================================== INPUT =================================
tau = 0.1;
C_vec = [1 10 100 1000];
%==========================================================================
lenC = length(C_vec);
tol = 1e-8;
n = 100;
p = 2;
q = 1;

%% Generate random data
Xy_train = Generate_random_data(n,p,q,1,0,1e-1);
X = Xy_train.X';
y = Xy_train.y;

index_pos = (y > 0);
index_neg = (y < 0);

X_pos = X(index_pos,:);
X_neg = X(index_neg,:);
xx_pos = X_pos(:,1);
yy_pos = X_pos(:,2);
xx_neg = X_neg(:,1);
yy_neg = X_neg(:,2);
x1 = [-0.4;0.4];

OPTIONS.tol = tol;
OPTIONS.tol_admm = 1e-4;
OPTIONS.ifrandom = 1;
OPTIONS.stop = 0;
OPTIONS.n = Xy_train.n;
OPTIONS.p = Xy_train.p;
OPTIONS.q = Xy_train.q;
OPTIONS.ifrandom = 1;
OPTIONS.flag_scaling = 0;
OPTIONS.sigmaiter = 2;
OPTIONS.sigmascale = 1.2;
OPTIONS.Test_sigma = 1;
OPTIONS.warm = 0;

OPTIONS.sigma = 10;
OPTIONS.sigalm_scale1 = 1;
OPTIONS.sigalm_scale2 = 2;
OPTIONS.sigalm_scale3 = 10;
OPTIONS.tau = tau;

%% Calculate the SM, ASM, and NSM for different values of C, and plot Figure 1.
for ii = 1:lenC
    OPTIONS.C = C_vec(ii);

    [~,W,b,runhist,info] = ALMSNCG(Xy_train.X,Xy_train.y,OPTIONS);
    lam = info.lam;
    

    w1 = W(1); w2 = W(2);
    eval(['Fig_h(',num2str(ii),') = subplot(2,4,',num2str(2*ii-1),');']); 

    % positive examples
    hold on
    h1 = plot(xx_pos, yy_pos, 'ro','MarkerSize',10,'LineWidth',1);

    % negative examples
    hold on
    h2 = plot(xx_neg, yy_neg, 'bd','MarkerSize',10,'LineWidth',1);

    % hyperplane H
    x_H = -(b + w1.*x1)./w2;
    hold on
    h3 = plot(x1,x_H,'-k','LineWidth',3);

    % hyperplane H_+
    x_Hpos = (1 - b - w1.*x1)./w2;
    hold on
    h4 = plot(x1,x_Hpos,'--k','LineWidth',3);

    % hyperplane H_neg
    x_Hneg = -(1 + b + w1.*x1)./w2;
    hold on
    h5 = plot(x1,x_Hneg,'--k','LineWidth',3);


    % index for SMM
    index_SMM = (-lam > tol*(1+sqrt(n))); %(xi > -tol*(1+sqrt(n)));    
    index_SMM_pos = index_SMM & index_pos;
    index_SMM_neg = index_SMM & index_neg;
    X_smm_pos = X(index_SMM_pos,:);
    X_smm_neg = X(index_SMM_neg,:);

    xx_smm_pos = X_smm_pos(:,1);
    yy_smm_pos = X_smm_pos(:,2);
    xx_smm_neg = X_smm_neg(:,1);
    yy_smm_neg = X_smm_neg(:,2);

    hold on
    plot(xx_smm_pos,yy_smm_pos,'ro','MarkerSize',10,'LineWidth',1,'MarkerFaceColor','r');
    hold on
    plot(xx_smm_neg,yy_smm_neg,'bd','MarkerSize',10,'LineWidth',1,'MarkerFaceColor','b');

    % indexJ
    index_ASM = (-lam > tol*(1+sqrt(n))) & (-lam < OPTIONS.C - tol*(1+sqrt(n)));
    X_J = X(index_ASM,:);
    xx_J = X_J(:,1);
    yy_J = X_J(:,2);
    hold on
    h6 = plot(xx_J,yy_J,'gs','MarkerSize',20,'LineWidth',2);
    xlabel('X(1)','FontSize',12);
    ylabel('X(2)','FontSize',12);

     num_SM = sum(index_SMM);
     num_ASM = sum(index_ASM);
     num_SM_rem_ASM = num_SM-num_ASM;
     num_NSM = n - num_SM;


     eval(['Fig_h(',num2str(ii),') = subplot(2,4,',num2str(2*ii),');']);

     % Percentage of SM and ASM
     explode = [1 0 0];
     pie_tem = pie([num_ASM, num_SM_rem_ASM, num_NSM],explode);

     pText = findobj(pie_tem,'Type','text');
     percentValues = get(pText,'String');
     txt = {'$\frac{|ASM|}{n}$';'$\frac{|SM/ASM|}{n}$';'$\frac{|NSM|}{n}$'};
     combinedtxt = strcat(txt,percentValues);

     % Merge LaTeX labels and percentage values
     for i = 1:length(pText)
         percent = strrep(percentValues{i}, '%', '\%'); 
         combinedtxt = sprintf('%s=%s', txt{i}, percent); 
         set(pText(i), 'String', combinedtxt, 'Interpreter', 'latex', 'FontSize', 16); 
     end

     map = [0.13333 0.5451 0.13333; 0.98039 0.50196 0.44706;1 0.84314 0];
     colormap(map)
     text_handles = pie_tem(2:2:end);

     % Adjust the label position
     for j = 1:length(text_handles)
         pos = get(text_handles(j), 'Position'); 
         pos = pos*1.035;
         set(text_handles(j), 'Position', pos, 'FontSize', 15); 
     end
end

% Add title for each line and pie charts
annotation('textbox', [0.13, 0.88, 0.35, 0.1], 'String', 'Comparison of SM, ASM, and NSM for C=1', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 15);
annotation('textbox', [0.54, 0.88, 0.35, 0.1], 'String', 'Comparison of SM, ASM, and NSM for C=10',...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 15);
annotation('textbox', [0.13, 0.40, 0.35, 0.1], 'String', 'Comparison of SM, ASM, and NSM for C=100',...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 15);
annotation('textbox', [0.54, 0.40, 0.35, 0.1], 'String', 'Comparison of SM, ASM, and NSM for C=1000',...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 15);

% Add a common legend
Fig_per = legend([h1, h2, h3, h4, h5, h6], 'positive examples','negative examples',...
    '$H:=\{X\,|\,\langle W,X \rangle=0\}$','$H_+:=\{X\,|\,\langle W,X \rangle=1\}$','$H_-:=\{X\,|\,\langle W,X \rangle=-1\}$','ASM');
set(Fig_per, 'FontSize', 15);
set(Fig_per,'Interpreter','latex','Units', 'normalized', 'Position', [0.5, 0.02, 0.0, 0.0], 'Orientation', 'horizontal');

%% Save Figure 1
set(gcf,'Units','normalized','Position',[0,0,1,1]);
exportgraphics(gcf,'SM_ASM.png','Resolution',300);











