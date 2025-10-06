function [freq, amp] = fft_basic(data, fs, same_unit, plot_figure)
% fft_basic - Computes and optionally plots the one-sided FFT of a time-domain signal.
%
% Syntax:
%   [freq, amp] = fft_basic(data, fs)
%   [freq, amp] = fft_basic(data, fs, same_unit, plot_figure)
%
% Inputs:
%   data        - Time-domain signal (vector)
%   fs          - Sampling frequency in Hz (scalar)
%   same_unit   - If set to 1 (default), scales the FFT amplitude to match the input signal's unit
%   plot_figure - If set to 1 (default), plots the FFT amplitude vs. frequency
%
% Outputs:
%   freq        - Frequency axis corresponding to the FFT (vector)
%   amp         - One-sided FFT amplitude spectrum (vector)
%
% Author: Jingyi Wu, 2025
    if nargin < 3
        same_unit = 1;
    end
    if nargin < 4
        plot_figure = 1;
    end

    M = length(data);
    NFFT = 2^nextpow2(M); % Zero-padding to next power of 2
    freq = fs/2 * linspace(0, 1, NFFT/2+1);

    func = data - mean(data);
    Y = fft(func, NFFT) / M;
    Y_half = Y(1:NFFT/2+1);
    
    amp = abs(Y_half);
    if same_unit == 1
        amp(2:end-1) = 2 * amp(2:end-1); % Double non-DC components
    end

    if plot_figure == 1
        figure;
        plot(freq, amp, 'k-');
        ylabel('FFT Amplitude');
        xlabel('Frequency (Hz)');
        title('FFT Result');
    end

end
