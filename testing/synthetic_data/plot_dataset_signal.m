clear
close all

%% load data
load('synthetic_dataset1.mat');
% load('synthetic_dataset2.mat');
% load('synthetic_dataset3.mat');

%% pick signal
idx_signal = 1;
signal_gt = squeeze(signal_test(idx_signal,:,2));
signal_noisy = squeeze(signal_test(idx_signal,:,1));
signal_raw = squeeze(signal_test(idx_signal,:,3));

%% setup path
addpath('../../data_generation/utils/');

%% Plot signals (Clean → Raw Noisy → Preprocessed)
x_pos = -0.045; 
y_pos = 1.16;
colors = colors();  % Custom color palette

figure; set(gcf,'Position',[0,0,600,600],'color','w'); hold on;
subplot1(3,1,'Gap',[0.02 0.055],'XTickL','Margin','YTickL','All');
apply_font_settings;

subplot1(1);
plot(time, signal_gt,'Color',colors{2},'LineWidth',2);
title('Clean Signal');
ylabel('Amplitude (a.u.)');
xlim([0 60]);
ylim([-1.5 1.5]);
yticks(-1:1:1);
label_figure('a', x_pos, y_pos);
box off
ax1 = gca;

subplot1(2);
plot(time, signal_raw,'Color',colors{4},'LineWidth',2);
title('Raw Noisy Signal');
ylabel('Amplitude (a.u.)');
xlim([0 60]);
label_figure('b', x_pos, y_pos);
box off
ax2 = gca;

subplot1(3);
plot(time, signal_noisy,'Color',colors{3},'LineWidth',2);
title('Preprocessed Noisy Signal');
ylabel('Amplitude (a.u.)');
xlabel('Time (s)','VerticalAlignment','cap');
xlim([0 60]);
label_figure('c', x_pos, y_pos);
box off
ax3 = gca;

linkaxes([ax1 ax2 ax3],'x');   % Synchronize x-axes