function filtered_signal = butterworth_basic(signal, fs, filter_type, low_bound, high_bound, order)
% butterworth_basic - Applies a Butterworth filter to the input signal.
%
% This function applies a low-pass, high-pass, or band-pass Butterworth filter
% using zero-phase filtering (`filtfilt`) to avoid phase distortion.
%
% Inputs:
%   signal       - Input time-domain signal (vector)
%   fs           - Sampling frequency in Hz (scalar)
%   filter_type  - Type of filter: 'low', 'high', or 'band'
%   low_bound    - Cutoff frequency for 'low'/'high' or lower bound for 'band' (Hz)
%   high_bound   - Upper bound for 'band' filter (Hz); not used for 'low' or 'high'
%   order        - Filter order (default = 3 if not provided)
%
% Output:
%   filtered_signal - Filtered output signal (same length as input)
%
% Author: Jingyi Wu, 2025

    if nargin < 6
        order = 3;
    end

    if strcmp(filter_type, 'low')
        [b, a] = butter(order, low_bound / (fs / 2), 'low');
    elseif strcmp(filter_type, 'high')
        [b, a] = butter(order, low_bound / (fs / 2), 'high');
    elseif strcmp(filter_type, 'band')
        [b, a] = butter(order, [low_bound high_bound] / (fs / 2));
    else
        disp('Filter type should be "low", "high", or "band".');
        return
    end

    filtered_signal = filtfilt(b, a, signal);
    disp('Signal filtered!');
end
