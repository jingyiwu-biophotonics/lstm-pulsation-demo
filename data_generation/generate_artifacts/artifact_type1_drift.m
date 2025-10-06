function [signal_noisy, drifts] = artifact_type1_drift(time, signal, num_drifts, plt_fig,...
    drift_locations, drift_amplitudes, drift_widths, drift_shapes, drift_directions)
    % This function generates baseline drifts modeled by skewed Gaussians.
    % Type 1 drifts rise or fall and return to baseline.
    %
    % Parameters:
    % - time: Time vector (n*1)
    % - signal: Original signal (n*1)
    % - num_drifts: Number of drifts
    % - plt_fig: Plot flag (1 to visualize, 0 to skip)
    % - drift_locations: Locations of drifts
    % - drift_amplitudes: Amplitudes of drifts
    % - drift_widths: Widths (scale) of drifts
    % - drift_shapes: Skewness of drifts
    % - drift_directions: Directions of drifts (+1 or -1)
    %
    % Outputs:
    % - signal_noisy: Signal with added baseline drifts
    % - drifts: The generated baseline drifts (n*1)
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
        num_drifts = randi([0 3]);
        plt_fig = 1;
        drift_locations = [];
        drift_amplitudes = [];
        drift_widths = [];
        drift_shapes = [];
        drift_directions = [];
    end

    if nargin < 3 || isempty(num_drifts)
        num_drifts = randi([0 3]); % Randomize number of drifts
    end

    if nargin < 4 || isempty(plt_fig)
        plt_fig = 1; % Default to plotting
    end

    %% Randomize Parameters if Not Provided
    if nargin < 5 || isempty(drift_locations)
        loc_min = 1;
        loc_max = length(time);
        drift_locations = sort(randi([loc_min, loc_max], num_drifts, 1)); % Random locations
    end

    if nargin < 6 || isempty(drift_amplitudes)
        amp_min = 3; amp_max = 30;
        drift_amplitudes = (amp_max - amp_min) * rand(num_drifts, 1) + amp_min; % Random amplitudes
    end

    if nargin < 7 || isempty(drift_widths)
        wid_min = 3; wid_max = 10; % Width in seconds
        drift_widths = (wid_max - wid_min) * rand(num_drifts, 1) + wid_min; % Random widths
    end

    if nargin < 8 || isempty(drift_shapes)
        shape_min = -15; shape_max = 15; % Skewness range
        drift_shapes = (shape_max - shape_min) * rand(num_drifts, 1) + shape_min; % Random skewness
    end

    if nargin < 9 || isempty(drift_directions)
        random_int = randi([1, 2], num_drifts, 1);
        drift_directions = 2 * random_int - 3; % Random directions (+1 or -1)
    end

    %% Generate Drifts
    drifts = zeros(length(time), 1);
    for idx_drift = 1:num_drifts
        amp = drift_amplitudes(idx_drift);
        wid = drift_widths(idx_drift);
        locn = time(drift_locations(idx_drift)); % Convert to time value
        shape = drift_shapes(idx_drift);
        direction = drift_directions(idx_drift);
        drift = direction * skewed_normal(time, locn, wid, shape, amp);
        drifts = drifts + drift;
    end
    
    signal_noisy = signal + drifts;

    %% Visualize
    if plt_fig == 1
        figure; set(gcf, 'Position', [100, 100, 600, 300]); hold on;
        plot(time, signal, 'Color', 'r', 'LineWidth', 1.5);
        s1 = plot(time, signal_noisy, 'Color', 'b', 'LineWidth', 1.5);
        s2 = plot(time, drifts, 'Color', 'k', 'LineWidth', 1.5);
        legend([s1, s2], {'Noisy signal','Drift'});
        title(['Type 1 Drift by Skewed Gaussian, Number of Drifts: ', num2str(num_drifts)]);
        ylabel('\DeltaOD (a.u.)');
        xlabel('Time (s)');
        xlim([0 max(time)]);
        box off;
    end
end

function sn = skewed_normal(x0,location,scale,shape,amplitude)
    % This function generates the probability density function for a skewed
    % normal distribution that is modified by some amplitude and scale.
    % It is based on: http://azzalini.stat.unipd.it/SN/
    
    x = (x0-location)./scale;
    sn = amplitude.*(2./scale).*normpdf(x).*normcdf(shape.*x);

end