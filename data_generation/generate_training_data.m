clear
close all

% Generate synthetic training data for the LSTM pulsation tracing model.
% For each chosen pulse type, we generate N pairs of:
%   - clean time traces
%   - noisy time traces (with multiple types of simulated artifacts)
%
% The final dataset is saved in a 3D array:
%   data [num_trials × num_timepoints × 3]
%   where channels = [preprocessed noisy signal, clean signal, raw noisy signal].
%
% Notes:
%   - In this example, 40,000 signals are generated per pulse type.
%   - Resulting datasets are about 2.5 GB in size (saved with -v7.3).
%   - Generation of 1-peak pulses is slower due to less efficient implementation.
%     Consider optimizing that function if generating large 1-peak datasets.
%
% Author: Jingyi Wu, 2025
%% Setup path
addpath(genpath(pwd));

%% Parameters
fs = 50;                              % Sampling frequency (Hz)
time = (0:1/fs:60-1/fs)';             % 60-second time axis
num_t = length(time);                 % Number of time samples per signal

plot_figure_pulse = 0;                % Toggle plotting of clean pulses
plot_figure_noise = 0;                % Toggle plotting of artifact waveforms
num_signal = 40000;                   % Number of signals to generate

% Select pulse type:
%   1 = closer double peak
%   2 = further double peak
%   3 = single peak
pulse_type = 1;

% Random seed for reproducibility
rng(1847365);
% rng(5637481); % This was used for "dataset_1peak_pulse_additional.mat"

% Pre-allocate arrays for efficiency
data_clean = zeros(num_t, num_signal);
data_noisy_raw = zeros(num_t, num_signal);

%% Random artifact parameters
% Each signal is perturbed by randomly parameterized artifact types
num_mod1 = randi(3, num_signal, 1);      % Gaussian modulation
num_mod2 = randi(3, num_signal, 1);      % Sigmoid modulation
num_ro   = randi(3, num_signal, 1);      % Random oscillation
num_spk  = randi(5, num_signal, 1);      % Spikes
num_s1   = randi(3, num_signal, 1);      % Shifts
num_d1   = randi(3, num_signal, 1);      % Drift type 1
num_d2   = randi(3, num_signal, 1);      % Drift type 2
num_pn   = randi([1,6], num_signal, 1);  % Perlin noise

%% Generate synthetic dataset
% Use parfor to speed up runtime
tic
parfor idx_signal = 1:num_signal
    % ----- Clean pulse -----
    [~, signal_clean] = generate_signal_with_quality_check(pulse_type, fs, plot_figure_pulse);

    % ----- Apply multiple artifacts sequentially -----
    signal_modulated = artifact_amplitude_modulation_gaussian(time, signal_clean, num_mod1(idx_signal), plot_figure_noise);
    signal_modulated = artifact_amplitude_modulation_sigmoid(time, signal_modulated, num_mod2(idx_signal), plot_figure_noise);
    [~, random_oscillation] = artifact_random_oscillation(time, signal_modulated, num_ro(idx_signal), plot_figure_noise);
    [~, spikes]            = artifact_spikes(time, signal_modulated, num_spk(idx_signal), plot_figure_noise);
    [~, shifts]            = artifact_shifts(time, signal_modulated, num_s1(idx_signal), plot_figure_noise);
    [~, drifts_sg]         = artifact_type1_drift(time, signal_modulated, num_d1(idx_signal), plot_figure_noise);
    [~, drifts_exp]        = artifact_type2_drift(time, signal_modulated, num_d2(idx_signal), plot_figure_noise);
    [~, perlin_noise]      = artifact_perlin_noise(time, signal_modulated, num_pn(idx_signal), plot_figure_noise);
    [~, colored_noise]     = artifact_colored_noise(time, signal_modulated, [], [], plot_figure_noise);

    % Combine all artifacts
    signal_noisy = signal_modulated + random_oscillation + spikes + ...
                   shifts + drifts_sg + drifts_exp + perlin_noise + colored_noise;

    % Save clean + noisy signals
    data_clean(:, idx_signal)     = signal_clean;
    data_noisy_raw(:, idx_signal) = signal_noisy;
end
toc

%% Process raw noisy signals (basic high-pass filter)
data_noisy_filt = butterworth_basic(data_noisy_raw, fs, 'high', 0.5, [], 5);

%% Package into final dataset
% Combine [filtered noisy, clean, raw noisy] into one matrix
data = cat(3, data_noisy_filt, data_clean, data_noisy_raw);
data = permute(data, [2,1,3]); % Reorder to: num_signals × num_time × num_channels

%% Save dataset
% Uncomment to save (approx 2.5 GB each):
% save('dataset_2peak_pulse_type1.mat', 'data', 'fs', 'time', '-v7.3');
% save('dataset_2peak_pulse_type2.mat', 'data', 'fs', 'time', '-v7.3');
% save('dataset_1peak_pulse.mat',       'data', 'fs', 'time', '-v7.3');

