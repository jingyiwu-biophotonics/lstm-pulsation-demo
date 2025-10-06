function [signal_noisy, colored_noise] = artifact_colored_noise(time, signal, alpha, amplitude, plt_fig)
    % This function generates colored noise using MATLAB's DSP System Toolbox.
    % The generated noise can be white, pink, or brownian depending on alpha.
    % https://www.mathworks.com/help/dsp/ref/dsp.colorednoise-system-object.html
    %
    % Inputs:
    % - time: Time vector
    % - signal: Original signal
    % - alpha: Noise power spectrum exponent (0=white, 1=pink, 2=brownian)
    % - amplitude: Noise amplitude scaling
    % - plt_fig: Plot flag (1 for visualization, 0 to skip)
    %
    % Example:
    % artifact_colored_noise;
    % artifact_colored_noise(time, signal, 0, 0.5, 1);
    %
    % Author: Jingyi Wu, 2025
    
    %% Ensure time and signal are column vectors
    if nargin > 0 && (~iscolumn(time) || ~iscolumn(signal))
        warning('Input "time" and "signal" must be column vectors. Reshaping...');
        time = time(:);
        signal = signal(:);
    end

    %% Example parameters
    if nargin == 0
        fs = 50; % Hz
        time = (0:1/fs:60-1/fs)'; % 60 seconds
        signal = sin(2*pi*1.2*time); % Example signal
        plt_fig = 1; % Enable plotting
    end
    
    if isempty(alpha) && isempty(amplitude)
        alpha_min = 0; alpha_max = 1.5;
        amp_min = 0.01; amp_max = 0.5;
        
        alpha = (alpha_max - alpha_min) * rand(1) + alpha_min; % Random alphas
        amplitude = (amp_max - amp_min) * rand(1) + amp_min; % Random amplitude scaling
    end

    if isempty(plt_fig)
        plt_fig = 1;
    end

    %% Generate colored noise
    nt = length(time); % Number of samples
    colored_noise_gen = dsp.ColoredNoise(alpha, nt, 1); % Create noise generator
    
    colored_noise = colored_noise_gen() * amplitude; % Scale noise
    signal_noisy = signal + colored_noise;

    %% Visualize
    if plt_fig == 1
        figure; set(gcf, 'Position', [100, 100, 600, 300]); hold on;
        plot(time, signal, 'Color', 'r', 'LineWidth', 1.5);
        s1 = plot(time, signal_noisy, 'Color', 'b', 'LineWidth', 1.5);
        legend(s1, 'Noisy signal');
        title(['Signal with Colored Noise, Alpha: ', num2str(alpha),', Amp.: ', num2str(amplitude)]);
        ylabel('\DeltaOD (a.u.)');
        xlabel('Time (s)');
        box off;
    end
end
