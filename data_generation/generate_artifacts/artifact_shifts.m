function [signal_noisy, shifts] = artifact_shifts(time, signal, num_shifts, plt_fig,...
    shift_loc_all, shift_slope_all, shift_amp_all, shift_dir_all)
    % This function generates baseline shifts using sigmoid functions with
    % randomized slopes, amplitudes, and directions.
    %
    % Parameters:
    % - time: Time vector (n*1)
    % - signal: Original signal (n*1)
    % - num_shifts: Number of shifts
    % - plt_fig: Plot flag (1 to visualize, 0 to skip)
    % - shift_loc_all: Locations of shifts
    % - shift_slope_all: Slopes of sigmoid functions
    % - shift_amp_all: Amplitudes of shifts
    % - shift_dir_all: Directions of shifts (+1 or -1)
    %
    % Outputs:
    % - signal_noisy: Signal with added baseline shifts
    % - shifts: The generated baseline shifts (n*1)
    %
    % Author: Jingyi Wu, 2025

    %% Ensure time and signal are column vectors
    if nargin > 0 && (~iscolumn(time) || ~iscolumn(signal))
        warning('Input "time" and "signal" must be column vectors. Reshaping...');
        time = time(:);
        signal = signal(:);
    end

    %% Sigmoid function for shifts
    fsig = @(x, a, b, c) c ./ (1 + exp(-a * (x - b)));

    %% Default Parameters
    if nargin == 0
        fs = 50; % Hz
        time = (0:1/fs:60-1/fs)'; % Time axis, 60 seconds
        signal = sin(2*pi*1.2*time); % Example signal (1.2 Hz sine wave)
        num_shifts = randi([0 3]);
        plt_fig = 1;
        shift_loc_all = [];
        shift_slope_all = [];
        shift_amp_all = [];
        shift_dir_all = [];
    end

    if nargin < 3 || isempty(num_shifts)
        num_shifts = randi([0 3]); % Randomize number of shifts
    end

    if nargin < 4 || isempty(plt_fig)
        plt_fig = 1; % Default to plotting
    end

    %% Randomize Parameters if Not Provided
    if nargin < 5 || isempty(shift_loc_all)
        loc_min = 1;
        loc_max = length(time);
        shift_loc_all = sort(randi([loc_min, loc_max], num_shifts, 1)); % Random locations
    end

    if nargin < 6 || isempty(shift_slope_all)
        slope_min = 1; slope_max = 5;
        shift_slope_all = slope_min + (slope_max - slope_min) * rand(num_shifts, 1); % Random slopes
    end

    if nargin < 7 || isempty(shift_amp_all)
        amp_min = 1.2; amp_max = 6;
        shift_amp_all = amp_min + (amp_max - amp_min) * rand(num_shifts, 1); % Random amplitudes
    end

    if nargin < 8 || isempty(shift_dir_all)
        shift_dir_all = randi([0, 1], num_shifts, 1) * 2 - 1; % Random directions (+1 or -1)
    end

    %% Generate Shifts
    shifts = zeros(length(time), 1);
    for idx_shift = 1:num_shifts
        slp = shift_slope_all(idx_shift); % Slope of the sigmoid
        locn = time(shift_loc_all(idx_shift)); % Center location of the shift
        amp = shift_amp_all(idx_shift); % Amplitude of the shift
        direction = shift_dir_all(idx_shift); % Direction (+1 or -1)

        if idx_shift == 1
            % First shift
            shift_n = direction * fsig(time, slp, locn, amp);
        else
            % Subsequent shifts build on previous shift's endpoint
            amp_before = shift_amp_all(idx_shift - 1);
            direction_before = shift_dir_all(idx_shift - 1);
            shift_n = direction * fsig(time, slp, locn, amp) + direction_before * amp_before;
        end

        % Add shift to cumulative signal
        shifts = shifts + shift_n;
    end

    signal_noisy = signal + shifts;

    %% Visualize
    if plt_fig == 1
        figure; set(gcf, 'Position', [100, 100, 600, 300]); hold on;
        plot(time, signal, 'Color', 'r', 'LineWidth', 0.8);
        s1 = plot(time, signal_noisy, 'Color', 'b', 'LineWidth', 1.5);
        s2 = plot(time, shifts, 'Color', 'k', 'LineWidth', 1.5);
        legend([s1, s2], {'Noisy signal','Shifts'});
        title(['Baseline Shifts, Number of Shifts: ', num2str(num_shifts)]);
        ylabel('\DeltaOD (a.u.)');
        xlabel('Time (s)');
        xlim([0 max(time)]);
        box off;
    end
end
