clc
clear
close all

% Import data -- uncomment datatset to be considered here
% dataset = 'breast-cancer-wisconsin';
dataset = 'iris';
% dataset = 'cgm';
% dataset = 'wu';
% dataset = 'wine';
% dataset = 'raisin';

try hmc_data_df = load(strjoin({'../results/hmc_data-df-',dataset,'.mat'},''));end
try hmc_data_dfi = load(strjoin({'../results/hmc_data-dfi-',dataset,'.mat'},''));end 
try dcc_data = load(strjoin({'../results/dcc_data-',dataset,'.mat'},''));end
try mcmc_data = load(strjoin({'../results/mcmc_data-',dataset,'.mat'},''));end
try smc_data = load(strjoin({'../results/smc_data-',dataset,'.mat'},''));end
try wu_data = load(strjoin({'../results/wu_data-',dataset,'.mat'},''));end

%% Plot figures for comparison
% BREAST CANCER WISCONSIN DATASET
if(strcmp(dataset,'breast-cancer-wisconsin'))
    figure('DefaultAxesFontSize',40)
    hmc_df_acc_mean = mean(reshape(hmc_data_df.hmc_df_acc_test,1000,10),2); 
    hmc_dfi_acc_mean = mean(reshape(hmc_data_dfi.hmc_dfi_acc_test,1000,10),2); 
    mcmc_acc_mean = mean(reshape(mcmc_data.mcmc_acc_test,1000,10),2); 
    wu_acc_mean = mean(reshape(wu_data.wu_acc_test,1000,10),2); 
    smc_acc_mean = mean(reshape(smc_data.smc_acc_test,1000,10),2); 
    x = 1:numel(hmc_df_acc_mean);
    hmc_df_acc_std = std(reshape(hmc_data_df.hmc_df_acc_test,1000,10),0,2);
    hmc_dfi_acc_std = std(reshape(hmc_data_dfi.hmc_dfi_acc_test,1000,10),0,2);
    mcmc_acc_std = std(reshape(mcmc_data.mcmc_acc_test,1000,10),0,2);
    wu_acc_std = std(reshape(wu_data.wu_acc_test,1000,10),0,2);
    smc_acc_std = std(reshape(smc_data.smc_acc_test,1000,10),0,2);
    plot(x, hmc_df_acc_mean,'Color', [0.49,0.18,0.56], 'LineWidth', 2);
    hold on
    plot(x, hmc_dfi_acc_mean, 'Color',[0.85,0.33,0.10],'LineWidth', 2);
    plot(x, wu_acc_mean, 'Color', [0.30,0.75,0.93], 'LineWidth', 2);
    plot(x, mcmc_acc_mean, 'Color',[0.47,0.67,0.19],'LineWidth', 2);
    plot(x, smc_acc_mean, 'Color',[0.93,0.69,0.13],'LineWidth', 2);
    legend("HMC-DF","HMC-DFI",'WU','CGM','SMC','fontsize',40,'Interpreter','latex','Location','SouthEast')
    curve1 = hmc_df_acc_mean + hmc_df_acc_std;
    curve2 = hmc_df_acc_mean - hmc_df_acc_std;
    x2 = [x, fliplr(x)];
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.49,0.18,0.56],'FaceAlpha',0.2 );
    hold on;
    curve1 = wu_acc_mean + wu_acc_std;
    curve2 = wu_acc_mean - wu_acc_std;
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.30,0.75,0.93],'FaceAlpha',0.2);
    curve1 = hmc_dfi_acc_mean + hmc_dfi_acc_std;
    curve2 = hmc_dfi_acc_mean - hmc_dfi_acc_std;
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.85,0.33,0.10],'FaceAlpha',0.2 );
    curve1 = mcmc_acc_mean + mcmc_acc_std;
    curve2 = mcmc_acc_mean - mcmc_acc_std;
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.47,0.67,0.19],'FaceAlpha',0.2 );
    curve1 = smc_acc_mean + smc_acc_std;
    curve2 = smc_acc_mean - smc_acc_std;
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.93,0.69,0.13],'FaceAlpha',0.2 );
    xlim([0,300])
    ylabel('Testing accuracy','fontsize',60,'Interpreter','latex')
    xlabel("Iteration",'fontsize',60,'Interpreter','latex')
    set(gca,'TickLabelInterpreter','latex')
end

% CGM DATASET
if(strcmp(dataset,'cgm'))
    figure('DefaultAxesFontSize',40)
    hmc_df_acc_mean = mean(reshape(hmc_data_df.hmc_df_acc_test,1000,10),2); 
    hmc_dfi_acc_mean = mean(reshape(hmc_data_dfi.hmc_dfi_acc_test,1000,10),2); 
    mcmc_acc_mean = mean(reshape(mcmc_data.mcmc_acc_test,5000,10),2); 
    wu_acc_mean = mean(reshape(wu_data.wu_acc_test,1000,10),2); 
    x = 1:numel(hmc_df_acc_mean);
    xmcmc = 1:numel(mcmc_acc_mean);
    hmc_df_acc_std = std(reshape(hmc_data_df.hmc_df_acc_test,1000,10),0,2);
    hmc_dfi_acc_std = std(reshape(hmc_data_dfi.hmc_dfi_acc_test,1000,10),0,2);
    mcmc_acc_std = std(reshape(mcmc_data.mcmc_acc_test,5000,10),0,2);
    wu_acc_std = std(reshape(wu_data.wu_acc_test,1000,10),0,2);
    plot(x, hmc_df_acc_mean,'Color', [0.49,0.18,0.56], 'LineWidth', 2);
    hold on
    plot(x, hmc_dfi_acc_mean, 'Color',[0.85,0.33,0.10],'LineWidth', 2);
    plot(x, wu_acc_mean, 'Color', [0.30,0.75,0.93], 'LineWidth', 2);
    plot(xmcmc, mcmc_acc_mean, 'Color',[0.47,0.67,0.19],'LineWidth', 2);
    legend("HMC-DF","HMC-DFI",'WU','CGM','fontsize',40,'Interpreter','latex','Location','SouthEast')
    curve1 = hmc_df_acc_mean + hmc_df_acc_std;
    curve2 = hmc_df_acc_mean - hmc_df_acc_std;
    x2 = [x, fliplr(x)];
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.49,0.18,0.56],'FaceAlpha',0.2 );
    hold on;
    curve1 = wu_acc_mean + wu_acc_std;
    curve2 = wu_acc_mean - wu_acc_std;
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.30,0.75,0.93],'FaceAlpha',0.2);
    curve1 = hmc_dfi_acc_mean + hmc_dfi_acc_std;
    curve2 = hmc_dfi_acc_mean - hmc_dfi_acc_std;
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.85,0.33,0.10],'FaceAlpha',0.2 );
    x = 1:numel(mcmc_acc_mean);
    x2 = [x, fliplr(x)];
    curve1 = mcmc_acc_mean + mcmc_acc_std;
    curve2 = mcmc_acc_mean - mcmc_acc_std;
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.47,0.67,0.19],'FaceAlpha',0.2 );
    xlim([0,1000])
    ylabel('Testing MSE','fontsize',60,'Interpreter','latex')
    xlabel("Per-proposal Iteration",'fontsize',60,'Interpreter','latex')
    set(gca,'TickLabelInterpreter','latex')
end

% IRIS DATASET
if(strcmp(dataset,'iris'))
    figure('DefaultAxesFontSize',40)
    hmc_df_acc_mean = mean(reshape(hmc_data_df.hmc_df_acc_test,1000,10),2); 
    hmc_dfi_acc_mean = mean(reshape(hmc_data_dfi.hmc_dfi_acc_test,1000,10),2); 
    mcmc_acc_mean = mean(reshape(mcmc_data.mcmc_acc_test,1000,10),2); 
    smc_acc_mean = mean(reshape(smc_data.smc_acc_test,1000,10),2); 
    x = 1:numel(hmc_df_acc_mean);
    hmc_df_acc_std = std(reshape(hmc_data_df.hmc_df_acc_test,1000,10),0,2);
    hmc_dfi_acc_std = std(reshape(hmc_data_dfi.hmc_dfi_acc_test,1000,10),0,2);
    mcmc_acc_std = std(reshape(mcmc_data.mcmc_acc_test,1000,10),0,2);
    smc_acc_std = std(reshape(smc_data.smc_acc_test,1000,10),0,2);
    plot(x, hmc_df_acc_mean,'Color', [0.49,0.18,0.56], 'LineWidth', 2);
    hold on
    plot(x, hmc_dfi_acc_mean, 'Color',[0.85,0.33,0.10],'LineWidth', 2);
    plot(x, mcmc_acc_mean, 'Color',[0.47,0.67,0.19],'LineWidth', 2);
    plot(x, smc_acc_mean, 'Color',[0.93,0.69,0.13],'LineWidth', 2);
    legend("HMC-DF","HMC-DFI",'CGM','SMC','fontsize',40,'Interpreter','latex','Location','SouthEast')
    curve1 = hmc_df_acc_mean + hmc_df_acc_std;
    curve2 = hmc_df_acc_mean - hmc_df_acc_std;
    x2 = [x, fliplr(x)];
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.49,0.18,0.56],'FaceAlpha',0.2 );
    hold on;
    curve1 = hmc_dfi_acc_mean + hmc_dfi_acc_std;
    curve2 = hmc_dfi_acc_mean - hmc_dfi_acc_std;
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.85,0.33,0.10],'FaceAlpha',0.2 );
    curve1 = mcmc_acc_mean + mcmc_acc_std;
    curve2 = mcmc_acc_mean - mcmc_acc_std;
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.47,0.67,0.19],'FaceAlpha',0.2 );
    curve1 = smc_acc_mean + smc_acc_std;
    curve2 = smc_acc_mean - smc_acc_std;
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.93,0.69,0.13],'FaceAlpha',0.2 );
    xlim([0,300])
    ylabel('Testing accuracy','fontsize',60,'Interpreter','latex')
    xlabel("Iteration",'fontsize',60,'Interpreter','latex')
    set(gca,'TickLabelInterpreter','latex')
end

% RAISIN DATASET
if(strcmp(dataset,'raisin'))
    figure('DefaultAxesFontSize',40)
    hmc_df_acc_mean = mean(reshape(hmc_data_df.hmc_df_acc_test,1000,10),2);     
    hmc_dfi_acc_mean = mean(reshape(hmc_data_dfi.hmc_dfi_acc_test,1000,10),2); 
    mcmc_acc_mean = mean(reshape(mcmc_data.mcmc_acc_test,1000,10),2); 
    wu_acc_mean = mean(reshape(wu_data.wu_acc_test,1000,10),2); 
    smc_acc_mean = mean(reshape(smc_data.smc_acc_test,1000,10),2); 
    x = 1:numel(hmc_df_acc_mean);
    hmc_df_acc_std = std(reshape(hmc_data_df.hmc_df_acc_test,1000,10),0,2);
    hmc_dfi_acc_std = std(reshape(hmc_data_dfi.hmc_dfi_acc_test,1000,10),0,2);
    mcmc_acc_std = std(reshape(mcmc_data.mcmc_acc_test,1000,10),0,2);
    wu_acc_std = std(reshape(wu_data.wu_acc_test,1000,10),0,2);
    smc_acc_std = std(reshape(smc_data.smc_acc_test,1000,10),0,2);
    plot(x, hmc_df_acc_mean,'Color', [0.49,0.18,0.56], 'LineWidth', 2);
    hold on
    plot(x, hmc_dfi_acc_mean, 'Color',[0.85,0.33,0.10],'LineWidth', 2);
    plot(x, wu_acc_mean, 'Color', [0.30,0.75,0.93], 'LineWidth', 2);
    plot(x, mcmc_acc_mean, 'Color',[0.47,0.67,0.19],'LineWidth', 2);
    plot(x, smc_acc_mean, 'Color',[0.93,0.69,0.13],'LineWidth', 2);
    legend("HMC-DF","HMC-DFI",'WU','CGM','SMC','fontsize',40,'Interpreter','latex','Location','SouthEast')
    curve1 = hmc_df_acc_mean + hmc_df_acc_std;
    curve2 = hmc_df_acc_mean - hmc_df_acc_std;
    x2 = [x, fliplr(x)];
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.49,0.18,0.56],'FaceAlpha',0.2 );
    hold on;
    curve1 = wu_acc_mean + wu_acc_std;
    curve2 = wu_acc_mean - wu_acc_std;
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.30,0.75,0.93],'FaceAlpha',0.2);
    curve1 = hmc_dfi_acc_mean + hmc_dfi_acc_std;
    curve2 = hmc_dfi_acc_mean - hmc_dfi_acc_std;
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.85,0.33,0.10],'FaceAlpha',0.2 );
    curve1 = mcmc_acc_mean + mcmc_acc_std;
    curve2 = mcmc_acc_mean - mcmc_acc_std;
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.47,0.67,0.19],'FaceAlpha',0.2 );
    curve1 = smc_acc_mean + smc_acc_std;
    curve2 = smc_acc_mean - smc_acc_std;
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.93,0.69,0.13],'FaceAlpha',0.2 );
    xlim([0,300])
    ylabel('Testing accuracy','fontsize',60,'Interpreter','latex')
    xlabel("Iteration",'fontsize',60,'Interpreter','latex')
    set(gca,'TickLabelInterpreter','latex')
end

% WINE DATASET
if(strcmp(dataset,'wine'))
    figure('DefaultAxesFontSize',40)
    hmc_df_acc_mean = mean(reshape(hmc_data_df.hmc_df_acc_test,1000,10),2); 
    hmc_dfi_acc_mean = mean(reshape(hmc_data_dfi.hmc_dfi_acc_test,1000,10),2); 
    mcmc_acc_mean = mean(reshape(mcmc_data.mcmc_acc_test,1000,10),2); 
    smc_acc_mean = mean(reshape(smc_data.smc_acc_test,1000,10),2); 
    x = 1:numel(hmc_df_acc_mean);
    hmc_df_acc_std = std(reshape(hmc_data_df.hmc_df_acc_test,1000,10),0,2);
    hmc_dfi_acc_std = std(reshape(hmc_data_dfi.hmc_dfi_acc_test,1000,10),0,2);
    mcmc_acc_std = std(reshape(mcmc_data.mcmc_acc_test,1000,10),0,2);
    smc_acc_std = std(reshape(smc_data.smc_acc_test,1000,10),0,2);
    plot(x, hmc_df_acc_mean,'Color', [0.49,0.18,0.56], 'LineWidth', 2);
    hold on
    plot(x, hmc_dfi_acc_mean, 'Color',[0.85,0.33,0.10],'LineWidth', 2);
    plot(x, mcmc_acc_mean, 'Color',[0.47,0.67,0.19],'LineWidth', 2);
    plot(x, smc_acc_mean, 'Color',[0.93,0.69,0.13],'LineWidth', 2);
    legend("HMC-DF","HMC-DFI",'CGM','SMC','fontsize',40,'Interpreter','latex','Location','SouthEast')
    curve1 = hmc_df_acc_mean + hmc_df_acc_std;
    curve2 = hmc_df_acc_mean - hmc_df_acc_std;
    x2 = [x, fliplr(x)];
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.49,0.18,0.56],'FaceAlpha',0.2 );
    hold on;    
    curve1 = hmc_dfi_acc_mean + hmc_dfi_acc_std;
    curve2 = hmc_dfi_acc_mean - hmc_dfi_acc_std;
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.85,0.33,0.10],'FaceAlpha',0.2 );
    curve1 = mcmc_acc_mean + mcmc_acc_std;
    curve2 = mcmc_acc_mean - mcmc_acc_std;
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.47,0.67,0.19],'FaceAlpha',0.2 );
    curve1 = smc_acc_mean + smc_acc_std;
    curve2 = smc_acc_mean - smc_acc_std;
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.93,0.69,0.13],'FaceAlpha',0.2 );
    xlim([0,300])
    ylabel('Testing accuracy','fontsize',60,'Interpreter','latex')
    xlabel("Iteration",'fontsize',60,'Interpreter','latex')
    set(gca,'TickLabelInterpreter','latex')
end

% LARGE-REAL DATASET
if(strcmp(dataset,'large-real'))
    figure('DefaultAxesFontSize',40)
    hmc_df_acc_mean = mean(reshape(hmc_data_df.hmc_df_acc_test,1000,10),2); 
    hmc_dfi_acc_mean = mean(reshape(hmc_data_dfi.hmc_dfi_acc_test,1000,10),2); 
    mcmc_acc_mean = mean(reshape(mcmc_data.mcmc_acc_test,50000,10),2); 
    wu_acc_mean = mean(reshape(wu_data.wu_acc_test,1000,10),2); 
    x = 1:numel(hmc_df_acc_mean);
    xmcmc = 1:numel(mcmc_acc_mean);
    hmc_df_acc_std = std(reshape(hmc_data_df.hmc_df_acc_test,1000,10),0,2);
    hmc_dfi_acc_std = std(reshape(hmc_data_dfi.hmc_dfi_acc_test,1000,10),0,2);
    mcmc_acc_std = std(reshape(mcmc_data.mcmc_acc_test,50000,10),0,2);
    wu_acc_std = std(reshape(wu_data.wu_acc_test,1000,10),0,2);
    plot(x, hmc_df_acc_mean,'Color', [0.49,0.18,0.56], 'LineWidth', 2);
    hold on
    plot(x, hmc_dfi_acc_mean, 'Color',[0.85,0.33,0.10],'LineWidth', 2);
    plot(x, wu_acc_mean, 'Color', [0.30,0.75,0.93], 'LineWidth', 2);
    plot(xmcmc, mcmc_acc_mean, 'Color',[0.47,0.67,0.19],'LineWidth', 2);
    legend("HMC-DF",'HMC-DFI','WU','CGM','fontsize',40,'Interpreter','latex','Location','SouthEast')
    curve1 = hmc_df_acc_mean + hmc_df_acc_std;
    curve2 = hmc_df_acc_mean - hmc_df_acc_std;
    x2 = [x, fliplr(x)];
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.49,0.18,0.56],'FaceAlpha',0.2 );
    hold on;    
    curve1 = hmc_dfi_acc_mean + hmc_dfi_acc_std;
    curve2 = hmc_dfi_acc_mean - hmc_dfi_acc_std;
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.85,0.33,0.10],'FaceAlpha',0.2 );
    curve1 = wu_acc_mean + wu_acc_std;
    curve2 = wu_acc_mean - wu_acc_std;
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.30,0.75,0.93],'FaceAlpha',0.2);
    x = 1:numel(mcmc_acc_mean);
    x2 = [x, fliplr(x)];
    curve1 = mcmc_acc_mean + mcmc_acc_std;
    curve2 = mcmc_acc_mean - mcmc_acc_std;
    inBetween = [curve1', fliplr(curve2')];
    fill(x2, inBetween, [0.47,0.67,0.19],'FaceAlpha',0.2 );
    xlim([0,1000])
    ylabel('Testing accuracy','fontsize',60,'Interpreter','latex')
    xlabel("Iteration",'fontsize',60,'Interpreter','latex')
    set(gca,'TickLabelInterpreter','latex')
end

%% Print out info for tables 
% clc

% Define indicies for calculating statistics - remove burn-in
indx = [501:1000,1501:2000,2501:3000,3501:4000,4501:5000,5501:6000,6501:7000,7501:8000,8501:9000,9501:10000]; % hmc, wu
indx_cgm = [4501:5000,9501:10000,14501:15000,19501:20000,24501:25000,29501:30000,34501:35000,39501:40000,44501:45000,49501:50000]; % cgm on cgm
indx_lr = [49501:50000,99501:100000,149501:150000,199501:200000,249501:250000,299501:300000,349501:350000,399501:400000,449501:450000,499501:500000]; % cgm on large-real
indx_smc = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000];

% Testing accuracy
mean_hmc_df = mean(hmc_data_df.hmc_df_acc_test(indx));
std_hmc_df = std(hmc_data_df.hmc_df_acc_test(indx));
mean_dcc = mean(dcc_data.dcc_acc_test);
std_dcc = std(dcc_data.dcc_acc_test);
if(strcmp(dataset,'cgm'))
    mean_smc=-1;std_smc=-1;
    mean_cgm = mean(mcmc_data.mcmc_acc_test(indx_cgm));std_cgm = std(mcmc_data.mcmc_acc_test(indx_cgm));
    mean_hmc_dfi = mean(hmc_data_dfi.hmc_dfi_acc_test(indx));std_hmc_dfi = std(hmc_data_dfi.hmc_dfi_acc_test(indx));
elseif(strcmp(dataset,'large-real'))
    mean_smc=-1;std_smc=-1;
    mean_cgm = mean(mcmc_data.mcmc_acc_test(indx_lr));std_cgm = std(mcmc_data.mcmc_acc_test(indx_lr));
    mean_hmc_dfi = mean(hmc_data_dfi.hmc_dfi_acc_test(indx));std_hmc_dfi = std(hmc_data_dfi.hmc_dfi_acc_test(indx));
elseif(strcmp(dataset,'wu'))
    mean_smc=-1;std_smc=-1;
    mean_cgm = mean(mcmc_data.mcmc_acc_test(indx));std_cgm = std(mcmc_data.mcmc_acc_test(indx));
    mean_hmc_dfi = mean(hmc_data_dfi.hmc_dfi_acc_test(indx));std_hmc_dfi = std(hmc_data_dfi.hmc_dfi_acc_test(indx));
else
    mean_smc=mean(smc_data.smc_acc_test(indx_smc));std_smc=std(smc_data.smc_acc_test(indx_smc));
    mean_cgm = mean(mcmc_data.mcmc_acc_test(indx));std_cgm = std(mcmc_data.mcmc_acc_test(indx));
    mean_hmc_dfi = mean(hmc_data_dfi.hmc_dfi_acc_test(indx));std_hmc_dfi = std(hmc_data_dfi.hmc_dfi_acc_test(indx));
end
if(strcmp(dataset,'iris')||strcmp(dataset,'wine'));mean_wu=-1;std_wu=-1;else;mean_wu=mean(wu_data.wu_acc_test(indx));std_wu=std(wu_data.wu_acc_test(indx));end

% Training accuracy
mean_hmc_df_train = mean(hmc_data_df.hmc_df_acc_train(indx));
std_hmc_df_train = std(hmc_data_df.hmc_df_acc_train(indx));
mean_dcc_train = mean(dcc_data.dcc_acc_train);
std_dcc_train = std(dcc_data.dcc_acc_train);
if(strcmp(dataset,'cgm'))
    mean_smc_train=-1;std_smc_train=-1;
    mean_cgm_train = mean(mcmc_data.mcmc_acc_train(indx_cgm));std_cgm_train = std(mcmc_data.mcmc_acc_train(indx_cgm));
    mean_hmc_dfi_train = mean(hmc_data_dfi.hmc_dfi_acc_train(indx));std_hmc_dfi_train = std(hmc_data_dfi.hmc_dfi_acc_train(indx));
elseif(strcmp(dataset,'large-real'))
    mean_smc_train=-1;std_smc_train=-1;
    mean_cgm_train = mean(mcmc_data.mcmc_acc_train(indx_lr));std_cgm_train = std(mcmc_data.mcmc_acc_train(indx_lr));
    mean_hmc_dfi_train = mean(hmc_data_dfi.hmc_dfi_acc_train(indx));std_hmc_dfi_train = std(hmc_data_dfi.hmc_dfi_acc_train(indx));
elseif(strcmp(dataset,'wu'))
    mean_smc_train=-1;std_smc_train=-1;
    mean_cgm_train = mean(mcmc_data.mcmc_acc_train(indx));std_cgm_train = std(mcmc_data.mcmc_acc_train(indx));
    mean_hmc_dfi_train = mean(hmc_data_dfi.hmc_dfi_acc_train(indx));std_hmc_dfi_train = std(hmc_data_dfi.hmc_dfi_acc_train(indx));
else
    mean_smc_train=mean(smc_data.smc_acc_train(indx_smc));std_smc_train=std(smc_data.smc_acc_train(indx_smc));
    mean_cgm_train = mean(mcmc_data.mcmc_acc_train(indx));std_cgm_train = std(mcmc_data.mcmc_acc_train(indx));
    mean_hmc_dfi_train = mean(hmc_data_dfi.hmc_dfi_acc_train(indx));std_hmc_dfi_train = std(hmc_data_dfi.hmc_dfi_acc_train(indx));
end
if(strcmp(dataset,'iris')||strcmp(dataset,'wine'));mean_wu_train=-1;std_wu_train=-1;else;mean_wu_train=mean(wu_data.wu_acc_train(indx));std_wu_train=std(wu_data.wu_acc_train(indx));end

% Average number of unique input predictors
mean_nvars_hmc_df = mean(hmc_data_df.hmc_df_num_unique_feats(indx));
mean_nvars_dcc = -1;
if(strcmp(dataset,'cgm'))
    mean_nvars_smc = -1;
    mean_nvars_cgm = mean(mcmc_data.mcmc_num_unique_feats(indx_cgm));
    mean_nvars_hmc_dfi = mean(hmc_data_dfi.hmc_dfi_num_unique_feats(indx));
elseif(strcmp(dataset,'large-real'))
    mean_nvars_smc = -1;
    mean_nvars_cgm = mean(mcmc_data.mcmc_num_unique_feats(indx_lr));
    mean_nvars_hmc_dfi = mean(hmc_data_dfi.hmc_dfi_num_unique_feats(indx));
elseif(strcmp(dataset,'wu'))
    mean_nvars_smc = -1;
    mean_nvars_cgm = mean(mcmc_data.mcmc_num_unique_feats(indx));
    mean_nvars_hmc_dfi = mean(hmc_data_dfi.hmc_dfi_num_unique_feats(indx));    
else
    mean_nvars_smc = mean(smc_data.smc_num_unique_feats(indx_smc));
    mean_nvars_cgm = mean(mcmc_data.mcmc_num_unique_feats(indx));
    mean_nvars_hmc_dfi = mean(hmc_data_dfi.hmc_dfi_num_unique_feats(indx));
end
if(strcmp(dataset,'iris')||strcmp(dataset,'wine'));mean_nvars_wu = -1;else;mean_nvars_wu=mean(wu_data.wu_num_unique_feats(indx));end

% Average number of leaf/terminal nodes
mean_nleaves_hmc_df = mean(hmc_data_df.hmc_df_num_leaf_nodes(indx));
mean_nleaves_dcc = -1;
if(strcmp(dataset,'cgm'))
    mean_nleaves_smc = -1;
    mean_nleaves_cgm = mean(mcmc_data.mcmc_num_leaf_nodes(indx_cgm));
    mean_nleaves_hmc_dfi = mean(hmc_data_dfi.hmc_dfi_num_leaf_nodes(indx));
elseif(strcmp(dataset,'large-real'))
    mean_nleaves_smc = -1;
    mean_nleaves_cgm = mean(mcmc_data.mcmc_num_leaf_nodes(indx_lr));
    mean_nleaves_hmc_dfi = mean(hmc_data_dfi.hmc_dfi_num_leaf_nodes(indx));
elseif(strcmp(dataset,'wu'))
    mean_nleaves_smc = -1;
    mean_nleaves_cgm = mean(mcmc_data.mcmc_num_leaf_nodes(indx));
    mean_nleaves_hmc_dfi = mean(hmc_data_dfi.hmc_dfi_num_leaf_nodes(indx));
else
    mean_nleaves_smc = mean(smc_data.smc_num_leaf_nodes(indx_smc));
    mean_nleaves_cgm = mean(mcmc_data.mcmc_num_leaf_nodes(indx));
    mean_nleaves_hmc_dfi = mean(hmc_data_dfi.hmc_dfi_num_leaf_nodes(indx));
end
if(strcmp(dataset,'iris')||strcmp(dataset,'wine'));mean_nleaves_wu=-1;else;mean_nleaves_wu=mean(wu_data.wu_num_leaf_nodes(indx));end

% Average acceptance percentage
if(strcmp(dataset,'iris')||strcmp(dataset,'wine'))
    mean_percent_accepts = [mean(mcmc_data.mcmc_percent_accepts);mean(hmc_data_df.hmc_df_percent_accepts);mean(hmc_data_dfi.hmc_dfi_percent_accepts);-1;-1;-1];
else
    mean_percent_accepts = [mean(mcmc_data.mcmc_percent_accepts);mean(hmc_data_df.hmc_df_percent_accepts);mean(hmc_data_dfi.hmc_dfi_percent_accepts);-1;-1;mean(wu_data.wu_percent_accepts)];
    wu_mean_accepts_per_iter = mean(wu_data.wu_percent_accepts_per_iter)
end

% Tabulate results
method_names = ["CGM";"HMC-DF";"HMC-DFI";"DCC-Tree";"SMC";"Wu"];
meanTest = [mean_cgm;mean_hmc_df;mean_hmc_dfi;mean_dcc;mean_smc;mean_wu];
stdTest = [std_cgm;std_hmc_df;std_hmc_dfi;std_dcc;std_smc;std_wu];
meanTrain = [mean_cgm_train;mean_hmc_df_train;mean_hmc_dfi_train;mean_dcc_train;mean_smc_train;mean_wu_train];
stdTrain = [std_cgm_train;std_hmc_df_train;std_hmc_dfi_train;std_dcc_train;std_smc_train;std_wu_train];
mean_n_leaves = [mean_nleaves_cgm;mean_nleaves_hmc_df;mean_nleaves_hmc_dfi;mean_nleaves_dcc;mean_nleaves_smc;mean_nleaves_wu];
mean_n_vars = [mean_nvars_cgm;mean_nvars_hmc_df;mean_nvars_hmc_dfi;mean_nvars_dcc;mean_nvars_smc;mean_nvars_wu];
table(method_names,meanTrain,stdTrain,meanTest,stdTest,mean_n_leaves,mean_n_vars,mean_percent_accepts)

