function [signal_noisy, perlin_noise] = artifact_perlin_noise(time, signal, turns_number, plt_fig, ...
    turns_amplitude)
    % This function generates 1D Perlin noise as a baseline artifact. 
    % The noise adds realistic baseline fluctuations to the signal.
    % https://cs.nyu.edu/~perlin/noise/
    %
    % Parameters:
    % - time: Time vector (n*1)
    % - signal: Original signal (n*1)
    % - turns_number: Number of turns or fluctuations in the Perlin noise
    % - plt_fig: Plot flag (1 to visualize, 0 to skip)
    % - turns_amplitude: Amplitude of the Perlin noise
    %
    % Outputs:
    % - signal_noisy: Signal with added Perlin noise
    % - perlin_noise: The generated Perlin noise (n*1)
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
        turns_number = randi([2 6]);
        plt_fig = 1;
        turns_amplitude = (5 - 1) * rand(1) + 1; % Random amplitude
    end

    if nargin < 3 || isempty(turns_number)
        turns_number = randi([2 6]); % Default number of turns
    end

    if nargin < 4 || isempty(plt_fig)
        plt_fig = 1; % Default to plotting
    end

    if nargin < 5 || isempty(turns_amplitude)
        amp_min = 1; amp_max = 5;
        turns_amplitude = (amp_max - amp_min) * rand(1) + amp_min; % Random amplitude
    end
    
    %% Generate 1D Perlin Noise
    dx = 1 / 200; % Determines smoothness, may need adjustment based on signal length
    perlin_output = perlin_1d(turns_number, dx); % Generate Perlin noise
    x_new = linspace(0, time(end), length(perlin_output));
    perlin_noise = turns_amplitude * interp1(x_new, perlin_output, time);
    signal_noisy = signal + perlin_noise;
    
    %% Visualize
    if plt_fig == 1
        figure; set(gcf, 'Position', [100, 100, 600, 300]); hold on;
        plot(time, signal, 'Color', 'r', 'LineWidth', 1.5);
        s1 = plot(time, signal_noisy, 'Color', 'b', 'LineWidth', 1.5);
        s2 = plot(time, perlin_noise, 'Color', 'k', 'LineWidth', 1.5);
        legend([s1, s2], {'Noisy Signal', '1D Perlin Noise'});
        title(['1D Perlin Noise, Turns: ', num2str(turns_number)]);
        ylabel('\DeltaOD (a.u.)');
        xlabel('Time (s)');
        xlim([0 max(time)]);
        box off;
    end
end

