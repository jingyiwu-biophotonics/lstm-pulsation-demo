function [noisy_signal, spikes] = artifact_spikes(time, signal, num_spikes, plt_fig, ...
    spike_locations, spike_amplitudes, spike_widths, spike_directions)
    % This function generates spikes modeled by narrow Gaussians and adds them
    % to a given signal.
    %
    % Parameters:
    % - time: Time vector (n*1)
    % - signal: Original signal (n*1)
    % - num_spikes: Number of spikes
    % - plt_fig: Plotting flag (1 to enable, 0 to disable)
    % - spike_locations: Locations of spikes (optional)
    % - spike_amplitudes: Amplitudes of spikes (optional)
    % - spike_widths: Gaussian widths of spikes (optional)
    % - spike_directions: Directions of spikes (+1 for up, -1 for down, optional)
    %
    % Outputs:
    % - noisy_signal: Signal with added spikes
    % - spikes: The generated spikes (n*1)
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
        signal = sin(2*pi*1.2*time); % Example signal, sine wave at 1.2 Hz
        num_spikes = randi([0 4]);
        plt_fig = 1; % Default to plotting
    end

    if nargin < 3 || isempty(num_spikes)
        num_spikes = randi([0 4]);
    end

    if nargin < 4 || isempty(plt_fig)
        plt_fig = 1; % Default to plotting if not provided
    end

    %% Randomize Parameters if Not Provided
    if nargin < 5 || isempty(spike_locations)
        loc_min = 1;
        loc_max = length(time);
        spike_locations = randi([loc_min, loc_max], num_spikes, 1); % Random locations
    end

    if nargin < 6 || isempty(spike_amplitudes)
        amp_min = 1.5; amp_max = 8;
        spike_amplitudes = (amp_max-amp_min) * rand(num_spikes, 1) + amp_min; % Random amplitudes
    end

    if nargin < 7 || isempty(spike_widths)
        wid_min = 0.0005; wid_max = 0.02;
        spike_widths = (wid_max-wid_min) * rand(num_spikes, 1) + wid_min; % Random widths
    end

    if nargin < 8 || isempty(spike_directions)
        random_int = randi([1, 2], num_spikes, 1);
        spike_directions = 2 * random_int - 3; % Random directions (+1 or -1)
    end

    %% Sort Spike Locations
    spike_locations = sort(time(spike_locations));
    
    %% Generate Spikes
    spikes = zeros(length(time), 1);
    for idx_spk = 1:num_spikes
        amp = spike_amplitudes(idx_spk);
        wid = spike_widths(idx_spk);
        locn = spike_locations(idx_spk);
        dire = spike_directions(idx_spk);
        spike = dire * amp * exp(-(time-locn).^2 ./ (2*wid^2));
        spikes = spikes + spike;
    end
    
    noisy_signal = signal + spikes;

    %% Visualize
    if plt_fig == 1
        figure; set(gcf, 'Position', [100, 100, 600, 300]); hold on;
        plot(time, signal, 'Color', 'r', 'LineWidth', 1.5);
        s1 = plot(time, noisy_signal, 'Color', 'b', 'LineWidth', 1.5);
        legend(s1, 'Signal with spikes');
        title(['Signal with Spikes, Number of Spikes: ', num2str(num_spikes)]);
        ylabel('\DeltaOD (a.u.)');
        xlabel('Time (s)');
        xlim([0 max(time)]);
        box off;
    end
end
