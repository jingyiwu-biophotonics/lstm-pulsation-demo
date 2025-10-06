function signal_noisy = artifact_amplitude_modulation_gaussian(time, signal, num_artifact, plt_fig,...
    mod_len_all, mod_loc_all, mod_wid_all, mod_n_all, mod_amp_all, mod_dir_all)
    % This function generates sudden amplitude changes by multiplying the 
    % signal by Gaussian-like shapes.
    % 
    % Parameters:
    % - time: Time vector (n*1)
    % - signal: Original signal (n*1)
    % - num_artifact: Number of artifacts
    % - plt_fig: Flag for visualization (1 to plot, 0 to skip)
    % - mod_len_all: Durations of artifacts, unit: sample
    % - mod_loc_all: Locations of artifacts, unit: sample
    % - mod_wid_all: Widths of Gaussian-like modulations
    % - mod_n_all: Power values for Gaussian flatness
    % - mod_amp_all: Amplitudes of modulation
    % - mod_dir_all: Directions of modulation (+1 or -1)
    %
    % Author: Jingyi Wu, 2025

    %% Ensure time and signal are column vectors
    if nargin > 0 && (~iscolumn(time) || ~iscolumn(signal))
        warning('Input "time" and "signal" must be column vectors. Reshaping...');
        time = time(:);
        signal = signal(:);
    end

    %% Gaussian-like function for modulating the signal
    gaussian_modulation = @(x, a, b, n) exp(-((x - a) / b).^n); % Even n for flat apex

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
        len_min = floor(0.5 * fs + 1); len_max = floor(3 * fs); % Durations in samples
        mod_len_all = randi([len_min, len_max], num_artifact, 1);
    end

    if nargin < 6 || isempty(mod_loc_all)
        loc_min = 1; loc_max = length(time);
        mod_loc_all = randi([loc_min, loc_max], num_artifact, 1); % Random locations
    end

    if nargin < 7 || isempty(mod_wid_all)
        mod_wid_all = 1 + (3 - 1) * rand(num_artifact, 1); % Random widths
    end

    if nargin < 8 || isempty(mod_n_all)
        mod_n_all = randsample(2:2:6, num_artifact, true); % Random powers (even numbers)
    end

    if nargin < 9 || isempty(mod_amp_all)
        mod_amp_all = 0.5 + (1.5 - 0.5) * rand(num_artifact, 1); % Random amplitudes
    end

    if nargin < 10 || isempty(mod_dir_all)
        mod_dir_all = randi([0, 1], num_artifact, 1) * 2 - 1; % Random directions (+1 or -1)
    end

    %% Adjust Out-of-Bounds Artifacts
    artifact_end_locations = mod_len_all + mod_loc_all;
    idx_check = artifact_end_locations > length(time);
    mod_loc_all(idx_check) = mod_loc_all(idx_check) - (artifact_end_locations(idx_check) - length(time));
    
    %% Apply Modulations
    signal_noisy = signal;
    for idx_mod = 1:num_artifact
        t_mod_loc = mod_loc_all(idx_mod):mod_loc_all(idx_mod) + mod_len_all(idx_mod) - 1;
        t_mod = time(t_mod_loc);
        mod_center = (t_mod(1) + t_mod(end)) / 2; % Center without rounding
        mod_amp = mod_amp_all(idx_mod);
        mod_wid = mod_wid_all(idx_mod);
        mod_n = mod_n_all(idx_mod);
        mod_dir = mod_dir_all(idx_mod);

        if mod_dir == 1
            modulation_factor = mod_amp * gaussian_modulation(t_mod, mod_center, mod_wid, mod_n);
        else
            modulation_factor = -mod_amp * gaussian_modulation(t_mod, mod_center, mod_wid, mod_n) + 1;
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
        title(['Pulse Modulation by Gaussians, Artifacts: ', num2str(num_artifact)]);
        ylabel('\DeltaOD (a.u.)');
        xlabel('Time (s)');
        box off;
    end
end
