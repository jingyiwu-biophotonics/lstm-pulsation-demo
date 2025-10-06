function signal_noisy = artifact_amplitude_modulation_sigmoid(time, signal, num_artifact, plt_fig,...
    mod_len_all, mod_loc_all, mod_slp_all, mod_amp_all, mod_dir_all)
    % This function generates sudden amplitude changes by modulating the
    % signal using sigmoid functions.
    % 
    % Parameters:
    % - time: Time vector (n*1)
    % - signal: Original signal (n*1)
    % - num_artifact: Number of artifacts
    % - plt_fig: Plot flag (1 for visualization, 0 to skip)
    % - mod_len_all: Durations of artifacts, unit: sample
    % - mod_loc_all: Locations of artifacts, unit: sample
    % - mod_slp_all: Slopes of sigmoid modulations
    % - mod_amp_all: Amplitudes of modulations
    % - mod_dir_all: Directions of modulation (+1 or -1)
    %
    % Author: Jingyi Wu, 2025

    %% Ensure time and signal are column vectors
    if nargin > 0 && (~iscolumn(time) || ~iscolumn(signal))
        warning('Input "time" and "signal" must be column vectors. Reshaping...');
        time = time(:);
        signal = signal(:);
    end

    %% Sigmoid function for modulation
    sigmoid_modulation = @(x, a, b) 1 ./ (1 + exp(-a * (x - b))); % Sigmoid function

    %% Default Parameters
    if nargin == 0
        fs = 50; % Hz
        time = (0:1/fs:60-1/fs)'; % Time axis, 60 seconds
        signal = sin(2*pi*1.2*time); % Example signal (1.2 Hz sine wave)
        num_artifact = randi([0 3]);
        plt_fig = 1; % Default to plotting
    end

    if nargin < 3 || isempty(num_artifact)
        num_artifact = randi([0 3]); % Randomize the number of artifacts
    end

    if nargin < 4 || isempty(plt_fig)
        plt_fig = 1; % Default to plotting if not provided
    end

    %% Randomize Parameters if Not Provided
    if nargin < 5 || isempty(mod_len_all)
        fs = 1 / (time(2) - time(1)); % Sampling frequency
        len_min = floor(1 * fs + 1); len_max = floor(4 * fs); % Durations in samples
        mod_len_all = randi([len_min, len_max], num_artifact, 1);
    end

    if nargin < 6 || isempty(mod_loc_all)
        loc_min = 1; loc_max = length(time);
        mod_loc_all = randi([loc_min, loc_max], num_artifact, 1); % Random locations
    end

    if nargin < 7 || isempty(mod_slp_all)
        mod_slp_all = 0.5 + (2 - 0.5) * rand(num_artifact, 1); % Random slopes
    end

    if nargin < 8 || isempty(mod_amp_all)
        mod_amp_all = 0.25 + (1.5 - 0.25) * rand(num_artifact, 1); % Random amplitudes
    end

    if nargin < 9 || isempty(mod_dir_all)
        mod_dir_all = randi([0, 1], num_artifact, 1) * 2 - 1; % Random directions (+1 or -1)
    end

    %% Adjust Out-of-Bounds Artifacts
    artifact_end_locations = mod_len_all + mod_loc_all;
    idx_check = artifact_end_locations > length(time);
    mod_loc_all(idx_check) = mod_loc_all(idx_check) - (artifact_end_locations(idx_check) - length(time));
    
    %% Apply Sigmoid Modulations
    signal_noisy = signal;
    for idx_mod = 1:num_artifact
        t_mod_loc = mod_loc_all(idx_mod):mod_loc_all(idx_mod) + mod_len_all(idx_mod) - 1;
        t_mod = time(t_mod_loc);
        mod_center = (t_mod(1) + t_mod(end)) / 2; % Center without rounding
        mod_amp = mod_amp_all(idx_mod);
        mod_slope = mod_slp_all(idx_mod);
        mod_dir = mod_dir_all(idx_mod);

        if mod_dir == 1
            modulation_factor = mod_amp * sigmoid_modulation(t_mod, mod_slope, mod_center);
        else
            modulation_factor = -mod_amp * sigmoid_modulation(t_mod, mod_slope, mod_center) + 1;
        end 
        
        signal_noisy(t_mod_loc) = signal_noisy(t_mod_loc) .* modulation_factor;
    end

    %% Visualize
    if plt_fig == 1
        figure; set(gcf, 'Position', [100, 100, 600, 300]); hold on;
        plot(time, signal, 'Color', 'r', 'LineWidth', 1.5);
        s1 = plot(time, signal_noisy, 'Color', 'b', 'LineWidth', 1.5);
        if num_artifact ~= 0
            xline(time(mod_loc_all), 'k--', 'LineWidth', 1);
            legend(s1, 'Modulated Signal');
        else
            legend(s1, 'Modulated Signal');
        end
        title(['Pulse Modulation by Sigmoids, Artifacts: ', num2str(num_artifact)]);
        ylabel('\DeltaOD (a.u.)');
        xlabel('Time (s)');
        box off;
    end
end
