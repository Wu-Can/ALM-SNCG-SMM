%% Generate synthetic datasets following the process in Luo et al. (2015).
clc;clear;
for ii = 1:4
    
    switch ii
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
    [Xy_train,Xy_test] = Generate_random_data(n,p,q);
    
    eval(['save Xy_train_',num2str(n),'_',num2str(p),'_',num2str(q),'.mat Xy_train -v7.3']);
    eval(['save Xy_test_',num2str(n),'_',num2str(p),'_',num2str(q),'.mat Xy_test -v7.3']);
    
    clear Xy_train Xy_test
end
