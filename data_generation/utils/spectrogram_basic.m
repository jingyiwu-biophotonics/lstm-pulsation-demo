function [s, f, t, p] = spectrogram_basic(data, fs, segment_length_fraction, overlap_fraction, NFFT, filter_data, filter_type, low_bound, high_bound, order, plot_figure)
% spectrogram_basic - Computes and optionally plots the power spectrogram of a signal.
%
% This function computes a time-frequency representation of the input signal using the short-time
% Fourier transform (STFT). It supports optional band filtering before spectrogram calculation.
%
% Inputs:
%   data                  - Input time-domain signal (vector)
%   fs                    - Sampling frequency in Hz (scalar)
%   segment_length_fraction - Fraction of data used as window length for each STFT segment (e.g., 0.25)
%   overlap_fraction      - Fraction of each segment overlapped with the next (e.g., 0.9)
%   NFFT                  - Number of FFT points used in STFT (optional, default = next power of 2)
%   filter_data           - Set to 1 to apply Butterworth filter before STFT (default = 0)
%   filter_type           - Type of Butterworth filter: 'low', 'high', 'band'
%   low_bound             - Lower cutoff frequency in Hz for filtering
%   high_bound            - Upper cutoff frequency in Hz for filtering
%   order                 - Order of the Butterworth filter
%   plot_figure           - Set to 1 (default) to plot the spectrogram
%
% Outputs:
%   s  - Complex STFT matrix
%   f  - Frequency vector (Hz)
%   t  - Time vector (s)
%   p  - Power spectral density estimate (magnitude squared)
%
% Author: Jingyi Wu, 2025

    if nargin < 3 || isempty(segment_length_fraction)
        segment_length_fraction = 0.25;
    end
    if nargin < 4 || isempty(overlap_fraction)
        overlap_fraction = 0.9;
    end

    data_length = length(data);
    segment_length = floor(data_length * segment_length_fraction);
    overlap_length = floor(overlap_fraction * segment_length);

    if nargin < 5 || isempty(NFFT)
        NFFT = 2^nextpow2(data_length);
    end    
    
    if nargin < 6
        filter_data = 0;
    end

    if filter_data == 1
        data_in = butterworth_basic(data, fs, filter_type, low_bound, high_bound, order);
    else
        data_in = data;
    end
    
    [s, f, t, p] = spectrogram(data_in, segment_length, overlap_length, NFFT, fs, 'yaxis', 'power');

    if nargin < 11 || isempty(plot_figure)
        plot_figure = 1;
    end
    if plot_figure == 1
        figure; set(gcf, 'Position', [0, 0, 800, 300]);
        surf(t, f, 10*log10(p), 'EdgeColor', 'none');
        view(2);
        xlabel('Time (s)');
        ylabel('Frequency (Hz)');
        zlabel('Power (dB)');
        ylim([0 5]);
        c = colorbar;
        c.Label.String = 'Power (dB)';
        caxis('auto');
        grid off;
    end

end
