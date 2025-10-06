function [signal_noisy, random_oscillations] = artifact_random_oscillation(time, signal, num_artifact, plt_fig,...
    loc_all, len_all, num_sine, sine_amp_min, sine_amp_max, sine_freq_min, sine_freq_max)
    % This function generates random oscillations in the signal by 
    % combining sines at different frequencies and amplitudes.
    %
    % Parameters:
    % - time: Time vector (n*1)
    % - signal: Original signal (n*1)
    % - num_artifact: Number of oscillation artifacts
    % - plt_fig: Plot flag (1 to visualize, 0 to skip)
    % - loc_all: Starting locations of artifacts, unit: sample
    % - len_all: Durations of artifacts, unit: sample
    % - num_sine: Number of sine waves per artifact
    % - sine_amp_min, sine_amp_max: Amplitude range of sine waves
    % - sine_freq_min, sine_freq_max: Frequency range of sine waves
    %
    % Outputs:
    % - signal_noisy: Signal with added random oscillations
    % - random_oscillations: The generated random oscillations (n*1)
    %
    % Author: Jingyi Wu, 2025

    %% Ensure time and signal are column vectors
    if nargin > 0 && (~iscolumn(time) || ~iscolumn(signal))
        warning('Input "time" and "signal" must be column vectors. Reshaping...');
        time = time(:);
        signal = signal(:);
    end

    %% Default Parameters
    if nargin == 0
        fs = 50; % Hz
        time = (0:1/fs:60-1/fs)'; % Time axis, 60 seconds
        signal = sin(2*pi*1.2*time); % Example signal (1.2 Hz sine wave)
        num_artifact = randi([0 3]);
        plt_fig = 1;
        loc_all = [];
        len_all = [];
        num_sine = [];
        sine_amp_min = 0.1;
        sine_amp_max = 3;
        sine_freq_min = 0.1 * 2 * pi;
        sine_freq_max = 5 * 2 * pi;
    end

    if nargin < 3 || isempty(num_artifact)
        num_artifact = randi([0 3]); % Randomize the number of artifacts
    end

    if nargin < 4 || isempty(plt_fig)
        plt_fig = 1; % Default to plotting if not provided
    end

    %% Randomize Parameters if Not Provided
    if nargin < 5 || isempty(loc_all)
        loc_min = 1;
        loc_max = length(time);
        loc_all = randi([loc_min, loc_max], num_artifact, 1); % Random locations
    end

    if nargin < 6 || isempty(len_all)
        fs = 1 / (time(2) - time(1)); % Sampling frequency
        len_min = floor(0.5 * fs + 1); len_max = floor(5 * fs); % Duration range in samples
        len_all = randi([len_min, len_max], num_artifact, 1);
    end

    if nargin < 7 || isempty(num_sine)
        num_sine = randi([1 5], num_artifact, 1); % Random number of sines
    end

    if nargin < 8 || isempty(sine_amp_min)
        sine_amp_min = 0.1; % Default amplitude range
    end

    if nargin < 9 || isempty(sine_amp_max)
        sine_amp_max = 3;
    end

    if nargin < 10 || isempty(sine_freq_min)
        sine_freq_min = 0.1 * 2 * pi; % Default frequency range
    end

    if nargin < 11 || isempty(sine_freq_max)
        sine_freq_max = 5 * 2 * pi;
    end

    %% Adjust Out-of-Bounds Artifacts
    artifact_end_locations = len_all + loc_all;
    idx_check = artifact_end_locations > length(time);
    loc_all(idx_check) = loc_all(idx_check) - (artifact_end_locations(idx_check) - length(time));
    
    %% Generate Random Oscillations
    random_oscillations = zeros(length(time), 1);
    for i = 1:num_artifact
        t_loc = loc_all(i):loc_all(i) + len_all(i) - 1;
        t_rnd = time(t_loc);
        num_sines_here = num_sine(i);
        
        % Generate random sine parameters
        sine_amps = sine_amp_min + (sine_amp_max - sine_amp_min) * rand(num_sines_here, 1);
        sine_freqs = sine_freq_min + (sine_freq_max - sine_freq_min) * rand(num_sines_here, 1);
        
        % Sum sines to create the oscillation
        oscillation = zeros(length(t_rnd), 1);
        for j = 1:num_sines_here
            oscillation = oscillation + sine_amps(j) * sin(sine_freqs(j) * t_rnd);
        end
        
        % Add to the signal
        random_oscillations(t_loc) = oscillation;
    end
    
    signal_noisy = signal + random_oscillations;

    %% Visualize
    if plt_fig == 1
        figure; set(gcf, 'Position', [100, 100, 600, 300]); hold on;
        plot(time, signal, 'Color', 'r', 'LineWidth', 1.5);
        s1 = plot(time, signal_noisy, 'Color', 'b', 'LineWidth', 1.5);
        if num_artifact ~= 0
            xline(time(loc_all), 'k--', 'LineWidth', 1);
            legend(s1, 'Noisy Signal');
        else
            legend(s1, 'Noisy Signal');
        end
        title(['Random Oscillations, Artifacts: ', num2str(num_artifact)]);
        ylabel('\DeltaOD (a.u.)');
        xlabel('Time (s)');
        box off;
    end
end
