function [t_seg, s_seg] = generate_signal_with_quality_check(pulse_type, fs, plt_fig)
    % This function generates clean synthetic signal until some criteria
    % are satisfied. 
    % Example use:
    % [t_seg, s_seg] = generate_signal_with_quality_check(2, 50, 1);
    % See generate_signal.m for more details.
    
    % pulse type: 1, 2, or 3.
    % fs: sampling frequency in Hz.
    % n_pulse: approximate number of heart beats/pulses to generate.
    % plt_fig: 0 or 1, for visualizing the time trace.
    
    % Note:
    % This function keeps generating signal until the critera are met, so
    % when setting plt_fig to 1, we might end up getting lots of figures.
    %
    % Author: Jingyi Wu, 2025

    
    out = 0;
    while out == 0
        [t_seg, s_seg] = generate_signal(pulse_type, fs, plt_fig); % Generate the signal
        out = check_pulse_quality(s_seg, pulse_type, fs); % Check some criteria
    end
    
end

function out = check_respration_amp(s_seg, fs)
    % Criterion 1: Due to some random processes in the signal generation, 
    % we could end up with very large respiration amplitude in the signal, 
    % which is not good. This function checks if that happens.

    % This criteria should be check for all three types of pulse shape
    % see if there's very large peak at repiration frequency after 
    % performing Fourier transform.

    % Range for respiration frequency.
    % Watch out, resp_min, and resp_max are hard-coded!
    resp_min = 0.1; resp_max = 0.3; % Hz
    [freq, amp] = fft_basic(s_seg,fs,1,0);
    amp_max = max(amp(freq > resp_min & freq < resp_max));
    
    if amp_max > 0.11
        out = 0;
    else
        out = 1;
    end
end

function out = check_sys_dia_peak_heights(s_seg)
    % Criterion 2: Due to some random processes in the signal generation,
    % the distolic peak sometimes becomes higher than the systolic peak,
    % which is also not good. This function checks if that happens.

    % Note: this mainly happens when using type 1 pulse shape where we want
    % to make the systolic and diastolic peaks closer together.
    
    % find the starts and ends of all pulses
    s_seg_flip = -s_seg;
    [~,locs] = findpeaks(s_seg_flip,'MinPeakHeight',0.2);
    
    % find the starts and ends of one pulse
    pulse = s_seg(locs(1):locs(2));
    
    % find the systolic and diastolic peak
    [pks_pulse, ~] = findpeaks(pulse,'MinPeakHeight',0.1);
    
    n_peak = length(pks_pulse);
    if n_peak == 1
        out = 1;
    elseif n_peak == 2
        pk_sys = pks_pulse(1);
        pk_dia = pks_pulse(2);
        if pk_sys < pk_dia
            out = 0;
        else
            out = 1;
        end
    else
        % it happens
        out = 0;
    end
end

function out = check_dicrotic_notch(s_seg)
    % Criterion 3: this happens for type 2 pulse shape, where the systolic 
    % and diastolic peaks are farther apart. the dicrotic notch (valley 
    % between the systolic and diastolic peaks) in this case can drop too 
    % close to zero. we want to avoid it here but one can get rid of this 
    % criteria if needed.
    
    % first, we find the systolic peaks
    [~,locs_sys] = findpeaks(s_seg,'MinPeakHeight',0.4);
    
    % then get the waveform between two consecutive systolic peaks.
    pulse = s_seg(locs_sys(1):locs_sys(2));
    
    % flip this pulse, and find peaks again. we should only find the pulse
    % onset. however, if the dicrotic notch is too low, we will find two peaks.
    [pks_pulse, ~] = findpeaks(-pulse,'MinPeakHeight',0.65);
    
    n_peak = length(pks_pulse);
    if n_peak == 0 | n_peak == 1
        % sometimes this happens, but it's ok
        out = 1;
    elseif n_peak == 2
        out = 0;
    end
end

function out = check_pulse_quality(s_seg, pulse_type, fs)
    % This function checks if the criteria are met.
    
    out_c1 = check_respration_amp(s_seg, fs);

    if out_c1 == 0
        out = 0;
    else
        if pulse_type == 1
            out = check_sys_dia_peak_heights(s_seg);
        elseif pulse_type == 2
            out = check_dicrotic_notch(s_seg);
        elseif pulse_type == 3
            out = 1;
        end
    end
end
