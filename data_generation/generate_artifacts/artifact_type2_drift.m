function [signal_noisy, drifts] = artifact_type2_drift(time, signal, num_drifts, plt_fig,...
    drift_locations, lp1_all, lp2_all, ap1_all, ap2_all, drift_directions)
    % This function generates type 2 baseline drifts modeled by exponential
    % functions. Type 2 drifts can reach arbitrary magnitudes and do not 
    % necessarily return to baseline.
    %
    % Parameters:
    % - time: Time vector (n*1)
    % - signal: Original signal (n*1)
    % - num_drifts: Number of drifts
    % - plt_fig: Plot flag (1 to visualize, 0 to skip)
    % - drift_locations: Start locations of drifts
    % - lp1_all: Durations from t1 to t2 (samples)
    % - lp2_all: Durations from t2 to t3 (samples)
    % - ap1_all: Heights from y1 to y2
    % - ap2_all: Heights from y2 to y3
    % - drift_directions: Directions of drifts (+1 or -1)
    %
    % Outputs:
    % - signal_noisy: Signal with added type 2 drifts
    % - drifts: The generated type 2 drifts (n*1)
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
        lp1_all = [];
        lp2_all = [];
        ap1_all = [];
        ap2_all = [];
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
        drift_locations = sort(randi([loc_min, loc_max], num_drifts, 1)); % Random start locations
    end

    if nargin < 6 || isempty(lp1_all)
        lp1_min = 5; lp1_max = 15;
        lp1_all = (lp1_max - lp1_min) * rand(num_drifts, 1) + lp1_min; % Random t1-t2 durations
    end

    if nargin < 7 || isempty(lp2_all)
        lp2_min = 5; lp2_max = 15;
        lp2_all = (lp2_max - lp2_min) * rand(num_drifts, 1) + lp2_min; % Random t2-t3 durations
    end

    if nargin < 8 || isempty(ap1_all)
        ap1_min = 0.8; ap1_max = 10;
        ap1_all = (ap1_max - ap1_min) * rand(num_drifts, 1) + ap1_min; % Random y1-y2 heights
    end

    if nargin < 9 || isempty(ap2_all)
        mp2_min = 0.8; mp2_max = 2;
        ap2_all = ap1_all .* ((mp2_max - mp2_min) * rand(num_drifts, 1) + mp2_min); % Random y2-y3 heights
    end

    if nargin < 10 || isempty(drift_directions)
        random_int = randi([1, 2], num_drifts, 1);
        drift_directions = 2 * random_int - 3; % Random directions (+1 or -1)
    end

    %% Generate Type 2 Drifts
    drifts = zeros(length(time), 1);
    previous_ap2 = 0; % Initialize for cumulative adjustment
    for idx = 1:num_drifts
        lp1 = lp1_all(idx);
        lp2 = lp2_all(idx);
        ap1 = ap1_all(idx);
        ap2 = ap2_all(idx);
        direction = drift_directions(idx);

        % Generate drift using `baseline_drift`
        [~, s_out] = baseline_drift(time, lp1, lp2, ap1, ap2, direction);
        drift_length = length(s_out);

        % Define segment bounds
        i_t1 = drift_locations(idx);
        i_t2 = i_t1 + drift_length - 1;

        % Adjust drift and apply to signal
        s_out = s_out + previous_ap2; % Adjust for continuity
        if i_t2 <= length(time)
            drifts(i_t1:i_t2) = s_out;
            drifts(i_t2+1:end) = s_out(end); % New baseline after drift
        else
            overlap = i_t2 - length(time);
            drifts(i_t1:end) = s_out(1:end-overlap);
        end

        % Update the baseline for subsequent drifts
        previous_ap2 = s_out(end);
    end

    signal_noisy = signal + drifts;

    %% Visualize
    if plt_fig == 1
        figure; set(gcf, 'Position', [100, 100, 600, 300]); hold on;
        plot(time, signal, 'Color', 'r', 'LineWidth', 1.5);
        s1 = plot(time, signal_noisy, 'Color', 'b', 'LineWidth', 1.5);
        s2 = plot(time, drifts, 'Color', 'k', 'LineWidth', 1.5);
        legend([s1, s2], {'Noisy signal', 'Drift'});
        title(['Type 2 Drift by Exp., Number of Drifts: ', num2str(num_drifts)]);
        ylabel('\DeltaOD (a.u.)');
        xlabel('Time (s)');
        box off;
    end
end

function [t_out,s_out] = baseline_drift(t_seg,lp1,lp2,ap1,ap2,direction)
    % This function generates a baseline drift in signal.

    % (t1, y1), (t2, y2), and (t3, y3) are the start, mid, and end points
    % for the drift, respectively.

    % 'direction' determines if it increases then decreases, or decreases 
    % then increases.

    % The drift will be added directly to the existing signal, so we
    % also need its time axis 't'.

    %% Time points and heights    
    t1 = 0; t2 = t1+lp1; t3 = t2+lp2;
    y1 = 0; y2 = y1+ap1; y3 = y2-ap2;
    
    %% Part 1 of the drift: exp1 fit
    x_p1 = [t1,t2]';
    y_p10 = [y1,y2]'; dy_p1 = min(y_p10);
    y_p1 = y_p10-dy_p1; % make them >= 0
    
    opts = fitoptions('Method', 'NonlinearLeastSquares', 'StartPoint', [1 0]);
    exp_p1 = fit(x_p1,y_p1,'exp1',opts);
    
    p1_idx = t_seg >= t1 & t_seg < t2;
    t_p1 = t_seg(p1_idx);
    s_p1 = exp_p1(t_p1)+dy_p1;
    
    %% Part 2 of the drift: exp1 fit
    x_p2 = [t2,t3]';
    y_p20 = [y2,y3]'; dy_p2 = min(y_p20);
    y_p2 = y_p20-dy_p2; % make them >= 0
    
    exp_p2 = fit(x_p2,y_p2,'exp1',opts);
    
    p2_idx = t_seg >= t2 & t_seg < t3;
    t_p2 = t_seg(p2_idx);
    s_p2 = exp_p2(t_p2)+dy_p2;
    
    %% output
    s_out = direction*[s_p1;s_p2];
    t_out = [t_p1; t_p2];

end