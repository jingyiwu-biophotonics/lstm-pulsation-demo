clear
close all

% Demonstrate how specific types of artifacts/noise sources distort a clean signal. 
% Each section applies one artifact type and plots the clean vs. corrupted signal. 
%
% General pipeline:
%   1. Load an example clean signal.
%   2. Sequentially apply artifact functions (Gaussian modulation, random oscillation, spikes, etc.).
%   3. Plot comparisons of clean vs. artifact-distorted signals.
%
% Author: Jingyi Wu, 2025

%% Setup
addpath(genpath(pwd));
load('example_dataset.mat');
signal = squeeze(data(1,:,2))'; % Use one clean signal from the example dataset
plot_figure = 0;                % Toggle for plotting
colors = colors();              % Custom color palette
ItemTokenSize = [15, 15];       % Legend box size

%% 1. Gaussian Amplitude Modulation
mod_num = 2;
mod_len_all = [2.2*fs, 4*fs];
mod_loc_all = [120, 410];
mod_wid_all = [1,2];
mod_n_all = [2,2];
mod_amp_all = [3,0.95];
mod_dir_all = [1,-1];
signal_mod = artifact_amplitude_modulation_gaussian(time, signal, mod_num, plot_figure, mod_len_all, mod_loc_all, mod_wid_all, mod_n_all, mod_amp_all, mod_dir_all);

% paper figure
figure; set(gcf,'Position',[0,0,600,250],'color','w'); hold on;
plot(time, signal,'Color',colors{2},'LineWidth',2);
plot(time, signal_mod,'Color',colors{3},'LineWidth',2);
xlabel('Time (s)','VerticalAlignment','cap');
ylabel('Amplitude (a.u.)');


l = legend('Clean','Pulse modulation','box','off','NumColumns',2,'location','northeast');
l.ItemTokenSize = ItemTokenSize;
pos = get(l, 'Position');
pos(2) = pos(2) + 0.02;
set(l, 'Position', pos);

xlim([0 15]);
ylim([-3.1 3.1]);
box off
apply_font_settings;

%% 2. Random Oscillation
rng(123567);
num_ro = 1;
len_all = 3*fs;
loc_all = 300;
num_sine = 4;
sine_amp_min = 0.1; sine_amp_max = 2; % amplitude
sine_freq_min = 0.1*2*pi; sine_freq_max = 5*2*pi; % frequency
[signal_ro, random_oscillation] = artifact_random_oscillation(time, signal, num_ro, plot_figure, loc_all, len_all, num_sine, sine_amp_min, sine_amp_max, sine_freq_min, sine_freq_max);

% paper figure
figure; set(gcf,'Position',[0,0,600,250],'color','w'); hold on;
plot(time, signal,'Color',colors{2},'LineWidth',2);
plot(time, signal_ro,'Color',colors{3},'LineWidth',2);
xlabel('Time (s)','VerticalAlignment','cap');
ylabel('Amplitude (a.u.)');

l = legend('Clean','Random oscillation','box','off','NumColumns',2,'location','northeast');
l.ItemTokenSize = ItemTokenSize;
pos = get(l, 'Position');
pos(2) = pos(2) + 0.02;
set(l, 'Position', pos);

xlim([0 15]);
ylim([-4 4]);
box off
apply_font_settings;


%% 4. Spikes
spk_num = 3;
spk_loc = [100,350,520];
spk_amp = [7.5,3,4];
spk_width = [0.001,0.0005,0.02];
spk_direction = [1,-1,1];
[signal_spikes, spikes] = artifact_spikes(time, signal, spk_num, plot_figure,spk_loc, spk_amp, spk_width, spk_direction);

% paper figure
figure; set(gcf,'Position',[0,0,600,250],'color','w'); hold on;
plot(time, signal,'Color',colors{2},'LineWidth',2);
plot(time, signal_spikes,'Color',colors{3},'LineWidth',2);
xlabel('Time (s)','VerticalAlignment','cap');
ylabel('Amplitude (a.u.)');

l = legend('Clean','Spikes','box','off','NumColumns',2,'location','northeast');
l.ItemTokenSize = ItemTokenSize;
pos = get(l, 'Position');
pos(2) = pos(2) + 0.02;
set(l, 'Position', pos);

xlim([0 15]);
ylim([-4 8]);
yticks([-4,0,4,8]);
box off
apply_font_settings;

%% 5. Baseline Shift
shift_num = 1;
shift_loc = 300;
slope_min = 5; slope_max = 10;
shift_slp = 1.2;
shift_amp = 6;
shift_direction = 1;
[signal_shifts, shifts] = artifact_shifts(time, signal, shift_num, plot_figure, shift_loc, shift_slp, shift_amp, shift_direction);

% paper figure
figure; set(gcf,'Position',[0,0,600,250],'color','w'); hold on;
plot(time, signal,'Color',colors{2},'LineWidth',2);
plot(time, signal_shifts,'Color',colors{3},'LineWidth',2);
xlabel('Time (s)','VerticalAlignment','cap');
ylabel('Amplitude (a.u.)');

l = legend('Clean','Shift','box','off','NumColumns',2,'location','northeast');
l.ItemTokenSize = ItemTokenSize;
pos = get(l, 'Position');
pos(2) = pos(2) + 0.02;
set(l, 'Position', pos);

xlim([0 15]);
ylim([-2 9]);
yticks([0,4,8]);
box off
apply_font_settings;

%% 6. Baseline Drift, Skewed Gaussian
drift_num = 1;
drift_loc = 210;
drift_amp = 15;
drift_width = 3;
drift_shape = 4;
drift_direction = 1;
[signal_drifts_sg, drifts_sg] = artifact_type1_drift(time, signal, drift_num, plot_figure, drift_loc, drift_amp, drift_width, drift_shape, drift_direction);

% paper figure
figure; set(gcf,'Position',[0,0,600,250],'color','w'); hold on;
plot(time, signal,'Color',colors{2},'LineWidth',2);
plot(time, signal_drifts_sg,'Color',colors{3},'LineWidth',2);
xlabel('Time (s)','VerticalAlignment','cap');
ylabel('Amplitude (a.u.)');

l = legend('Clean','Drift, Skewed Gaussian','box','off','NumColumns',2,'location','northeast');
l.ItemTokenSize = ItemTokenSize;
pos = get(l, 'Position');
pos(2) = pos(2) + 0.02;
set(l, 'Position', pos);

xlim([0 15]);
ylim([-1.8 5]);
yticks(-2:2:6);
box off
apply_font_settings;

%% 7. Baseline Drift, Exponentials
exp_num = 1;
exp_loc = 1;
lp1_all = 8;
lp2_all = 4;
ap1_all = 8;
ap2_all = 4;
exp_direction = -1;
[signal_drifts_exp, drifts_exp] = artifact_type2_drift(time, signal, exp_num, plot_figure, exp_loc, lp1_all, lp2_all, ap1_all, ap2_all, exp_direction);

% paper figure
figure; set(gcf,'Position',[0,0,600,250],'color','w'); hold on;
plot(time, signal,'Color',colors{2},'LineWidth',2);
plot(time, signal_drifts_exp,'Color',colors{3},'LineWidth',2);
xlabel('Time (s)','VerticalAlignment','cap');
ylabel('Amplitude (a.u.)');

l = legend('Clean','Drift, Exponential','box','off','NumColumns',2,'location','northeast');
l.ItemTokenSize = ItemTokenSize;
pos = get(l, 'Position');
pos(2) = pos(2) + 0.02;
set(l, 'Position', pos);

xlim([0 15]);
ylim([-10 2.6]);
yticks([-8,-4,0]);
box off
apply_font_settings;

%% 8. Perlin Noise
rng(1231);
turns_num = 6;
turns_amp = 2.6;
[signal_perlin, perlin_noise] = artifact_perlin_noise(time, signal, turns_num, plot_figure, turns_amp);

% paper figure
figure; set(gcf,'Position',[0,0,600,250],'color','w'); hold on;
plot(time, signal,'Color',colors{2},'LineWidth',2);
plot(time, signal_perlin,'Color',colors{3},'LineWidth',2);
xlabel('Time (s)','VerticalAlignment','cap');
ylabel('Amplitude (a.u.)');

l = legend('Clean','Perlin noise','box','off','NumColumns',2,'location','northeast');
l.ItemTokenSize = ItemTokenSize;
pos = get(l, 'Position');
pos(2) = pos(2) + 0.02;
set(l, 'Position', pos);

xlim([0 60]);
ylim([-4 4]);
yticks(-4:2:4);
box off
apply_font_settings;

%% 9. Colored Noise
color_alpha = 1;
color_amp = 0.25;
[signal_color, colored_noise] = artifact_colored_noise(time, signal, color_alpha, color_amp, plot_figure);

% paper figure
figure; set(gcf,'Position',[0,0,600,250],'color','w'); hold on;
plot(time, signal,'Color',colors{2},'LineWidth',2);
plot(time, signal_color,'Color',colors{3},'LineWidth',2);
xlabel('Time (s)','VerticalAlignment','cap');
ylabel('Amplitude (a.u.)');

l = legend('Clean','Colored noise','box','off','NumColumns',2,'location','northeast');
l.ItemTokenSize = ItemTokenSize;
pos = get(l, 'Position');
pos(2) = pos(2) + 0.02;
set(l, 'Position', pos);

xlim([0 15]);
ylim([-2.2 2.2]);
yticks(-2:2:2);
box off
apply_font_settings;



