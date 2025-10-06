clear
close all

% This script demonstrates how to generate and visualize a single synthetic time trace.
% General pipeline:
%   1. Generate a clean pulse waveform.
%   2. Add sequential artifacts/noises to simulate real-world distortions.
%   3. Apply basic filtering to the noisy signal.
%   4. Plot the clean, raw noisy, and preprocessed signals.
%   5. Plot spectrograms for clean and noisy signals.
%
% Author: Jingyi Wu, 2025

%% Setup path
addpath(genpath(pwd));

%% Generate a clean pulse
pulse_type = randi(3);      % Randomly pick pulse type:
                            % 1 = closer double peak, 2 = further double peak, 3 = single peak
fs = 50;                    % Sampling frequency (Hz)
plot_figure_pulse = 0;      % Toggle plotting for pulse (can create many figures)

[time, signal] = generate_signal_with_quality_check(pulse_type, fs, plot_figure_pulse);

%% Generate artifacts and noise
plot_figure_noise = 0;      % Toggle plotting for artifact generation

% Each artifact type is added sequentially to distort the clean pulse
% 1. Gaussian Amplitude Modulation
num_mod1 = randi(3);
signal_modulated = artifact_amplitude_modulation_gaussian(time, signal, num_mod1, plot_figure_noise);

% 2. Sigmoid Amplitude Modulation
num_mod2 = randi(3);
signal_modulated = artifact_amplitude_modulation_sigmoid(time, signal_modulated, num_mod2, plot_figure_noise);

% 3. Random Oscillation
num_ro = randi(3);
[signal_noisy, random_oscillation] = artifact_random_oscillation(time, signal_modulated, num_ro, plot_figure_noise);

% 4. Spikes
num_spk = randi(5);
[signal_noisy, spikes] = artifact_spikes(time, signal_noisy, num_spk, plot_figure_noise);

% 5. Baseline Shift
num_s1 = randi(3);
[signal_noisy, shifts] = artifact_shifts(time, signal_noisy, num_s1, plot_figure_noise);

% 6. Baseline Drift, Skewed Gaussian
num_d1 = randi(3);
[signal_noisy, drifts_sg] = artifact_type1_drift(time, signal_noisy, num_d1, plot_figure_noise);

% 7. Baseline Drift, Exponentials
num_d2 = randi(3);
[signal_noisy, drifts_exp] = artifact_type2_drift(time, signal_noisy, num_d2, plot_figure_noise);

% 8. Perlin Noise
num_pn = randi([1 6]);
[signal_noisy, perlin_noise] = artifact_perlin_noise(time, signal_noisy, num_pn, plot_figure_noise);

% 9. Colored Noise
[signal_noisy, colored_noise] = artifact_colored_noise(time, signal_noisy, [], [], plot_figure_noise);

%% Process raw noisy signals (basic high-pass filter)
signal_filtered = butterworth_basic(signal_noisy, fs, 'high', 0.5, [], 5);

%% Plot signals (Clean → Raw Noisy → Preprocessed)
x_pos = -0.045; 
y_pos = 1.16;
colors = colors();  % Custom color palette

figure; set(gcf,'Position',[0,0,600,600],'color','w'); hold on;
subplot1(3,1,'Gap',[0.02 0.055],'XTickL','Margin','YTickL','All');
apply_font_settings;

subplot1(1);
plot(time, signal,'Color',colors{2},'LineWidth',2);
title('Clean Signal');
ylabel('Amplitude (a.u.)');
xlim([0 60]);
ylim([-1.5 1.5]);
yticks(-1:1:1);
label_figure('a', x_pos, y_pos);
box off
ax1 = gca;

subplot1(2);
plot(time, signal_noisy,'Color',colors{4},'LineWidth',2);
title('Raw Noisy Signal');
ylabel('Amplitude (a.u.)');
xlim([0 60]);
label_figure('b', x_pos, y_pos);
box off
ax2 = gca;

subplot1(3);
plot(time, signal_filtered,'Color',colors{3},'LineWidth',2);
title('Preprocessed Noisy Signal');
ylabel('Amplitude (a.u.)');
xlabel('Time (s)','VerticalAlignment','cap');
xlim([0 60]);
label_figure('c', x_pos, y_pos);
box off
ax3 = gca;

linkaxes([ax1 ax2 ax3],'x');   % Synchronize x-axes

%% Plot spectrograms
% FFT length set to ~half the nearest power of 2 of signal length
% NFFT = floor(2^nextpow2(length(signal))*0.5);
% 
% spectrogram_basic(signal, fs, 0.1, 0.9, NFFT, [], [], [], [], [], 1);
% title('Spectrogram for Clean Signal');
% clim([-150 10]);
% apply_font_settings;
% 
% spectrogram_basic(signal_noisy, fs, 0.1, 0.9, NFFT, [], [], [], [], [], 1);
% title('Spectrogram for Raw Noisy Signal');
% clim([-150 10]);
% apply_font_settings;
