function [t_seg, s_seg] = generate_signal(pulse_type, fs, plt_fig)
    %   This function generates a synthetic pulsatile time trace (clean baseline physiology)
    %   using a skewed-normal–based pulse model plus HR/RR dynamics adapted
    %   from ECGSYN. Returns a 60 s segment (by default) suitable for
    %   training data generation and demonstrations.
    %
    % Usage
    %   [t_seg, s_seg] = generate_signal(pulse_type, fs, plt_fig)
    %
    % Inputs
    %   pulse_type : 1, 2, or 3
    %                (1) closer double peak
    %                (2) farther double peak
    %                (3) single peak
    %   fs        : sampling frequency (Hz)
    %   plt_fig   : 0/1 — if 1, show diagnostic plots (clean signal, RR, spectrogram)
    %
    % Outputs
    %   t_seg     : time vector (seconds) for returned segment (column vector)
    %   s_seg     : signal segment (column vector), roughly scaled to [-1, 1]
    %
    % Notes
    %   • This function focuses on generating clean signal based on input
    %     physiological parameters.
    %   • The returned segment length is hard-coded below (see the block
    %     “final time segment”). If you change segment duration, also adjust
    %     related plotting/indices there.
    %   • Based on: https://physionet.org/content/ecgsyn/1.0.0/
    %
    % Author: Jingyi Wu, 2025

    % Example:
    % [t_seg, s_seg] = generate_signal(1, 50, 1);

    %% Pulse-shape parameter presets (skewed normal) for each pulse type
    % l*   = location; a* = amplitude scale; scal* = scale; sp* = shape
    % *_exp = expected center value, *_rg = uniform jitter range
    if pulse_type == 1
        % Type 1: systolic and diastolic peaks closer together
        l1_exp = 20; l2_exp = 97; l_rg = 4;
        a1_exp = 0.95; a2_exp = 1.5; a_rg = 0.1;
        scal1_exp = 1.2; scal2_exp = 0.9; scal_rg = 0.1;
        sp1_exp = 4.5; sp2_exp = 0; sp_rg = 0.25;
    
    elseif pulse_type == 2
        % Type 2: peaks farther apart
        l1_exp = 0; l2_exp = 86; l_rg = 3;
        a1_exp = 1; a2_exp = 1; a_rg = 0.1;
        scal1_exp = 1.1; scal2_exp = 0.9; scal_rg = 0.1;
        sp1_exp = 3; sp2_exp = 0; sp_rg = 0.2;
    
    elseif pulse_type == 3
        % Type 3: single peak
        l1_exp = 8; l2_exp = 8; l_rg = 4;
        a1_exp = 1; a2_exp = 1; a_rg = 0;
        scal1_exp = 5; scal2_exp = 5; scal_rg = 2;
        sp1_exp = 10; sp2_exp = 10; sp_rg = 8;

    end

    %% Randomize pulse parameters around their expected values for variabilites in morphology
    l1 = rand_uniform_centered(l1_exp,l_rg,1);
    l2 = rand_uniform_centered(l2_exp,l_rg,1);
    
    a1 = rand_uniform_centered(a1_exp,a_rg,1);
    a2 = rand_uniform_centered(a2_exp,a_rg,1);
    
    scal1 = rand_uniform_centered(scal1_exp,scal_rg,1);
    scal2 = rand_uniform_centered(scal2_exp,scal_rg,1);
    
    sp1 = rand_uniform_centered(sp1_exp,sp_rg,1);
    sp2 = rand_uniform_centered(sp2_exp,sp_rg,1);

    if pulse_type == 3
        location = [l1 l1];
        ai = [a1 a1];
        scale = [scal1 scal1];
        shape = [sp1 sp1];
    else
        location = [l1 l2];
        ai = [a1 a2];
        scale = [scal1 scal2];
        shape = [sp1 sp2];
    end
    
    %% Heart rate parameters
    % Minimum and maximum HR, in beats per minute
    hr_min = 40; hr_max = 150;
    hr_std = randi(8);

    hr_mean = randi([hr_min,hr_max],1);
    n_beats = ceil(hr_mean*1.2);
        
    %% Respiration and other parameters
    % Minimum and maximum respiration rate, in Hz
    resp_min = 0.1; resp_max = 0.3;
    f_resp = (resp_max-resp_min)*rand(1)+resp_min;
    f_resp_std = 0.01;
    
    % Respiration amplitude, arbitary unit
    amp_resp_min = 0.0001; amp_resp_max = 0.003;  
    amp_resp = (amp_resp_max-amp_resp_min)*rand(1)+amp_resp_min;
    
    % Mayer wave
    f_mayer = 0.1;
    f_mayer_std = 0.01;
    
    % low (Mayer wave) high (repiration) frequency ratio:
    ra_min = 0.4; ra_max = 0.6;
    lfhfratio = (ra_max-ra_min)*rand(1)+ra_min;
        
    %% Check that f_int is an integer multiple of fs 
    f_int = fs*2;
    q = round(f_int/fs);
    qd = f_int/fs;
    if q ~= qd
       error(['Internal sampling frequency (f_int) must be an integer multiple ' ... 
    'of the sampling frequency (fs). Your current choices are: ' ... 
    'fs = ' int2str(fs) ' and f_int = ' int2str(f_int) '.']);
    end
    
    %% Calculate time scales for rr and total output
    sampfreqrr = 1;
    trr = 1/sampfreqrr; 
    rrmean = 60/hr_mean;	 
    Nrr = 2^(ceil(log2(n_beats*rrmean/trr)));
    
    % compute rr process
    rr0 = rrprocess(f_mayer,f_resp,f_mayer_std,f_resp_std,lfhfratio,sampfreqrr,Nrr);
    
    % upsample rr time series from 1 Hz to sfint Hz
    rrup = interp(rr0,f_int);
    t_rrup = 0:1/f_int:length(rrup)/f_int-1/f_int;
    
    % hr time trace
    hr_rrup = hr_mean*ones(length(t_rrup),1);
    
    % add baseline
    rrstd = 60*hr_std./(hr_rrup.*hr_rrup);
    rr0std = std(rrup);
    ratio = rrstd/rr0std;
    rrmean_hr_rrup = 60./hr_rrup;
    rr = rrmean_hr_rrup + rrup.*ratio;
    
    % make the rrn time series
    dt = 1/f_int;
    rrn = zeros(length(rr),1);
    hr_rrn = zeros(length(rr),1);
    tnirs=0;
    i = 1;
    while i <= length(rr)
       tnirs = tnirs+rr(i);
       ip = round(tnirs/dt);
       rrn(i:ip) = rr(i);
       hr_rrn(i:ip) = hr_rrup(i);
       i = ip+1;
    end 
    Nt = ip;
    
    %% integrate system using fourth order Runge-Kutta
    location = location*pi/180; % convert to rad
    scale_hr = ones(size(hr_rrn)).*scale; % for every time point

    x0 = [1,0,0.04];
    Tspan = 0:dt:(Nt-1)*dt;
    drift_resp = 1; % include repiration drift
    [~, X0] = ode45('odenirs_2SN',Tspan,x0,[],rrn,f_int,ai,location,scale_hr,shape,f_resp,amp_resp,drift_resp);
    
    % downsample to required fs
    X = X0(1:q:end,:);
    
    %% Scale the output channel (X(:,3)) into roughly [-1, 1]
    V_max = 1;
    V_min = -1;
    z = X(:,3);
    
    % Take min/max over a mid-window (10–20 s) to avoid potential artifacts
    % at the beginning of the singal
    zmin = min(z(10*fs+1:20*fs));
    zmax = max(z(10*fs+1:20*fs));

    zrange = zmax - zmin;
    s_clean = (z - zmin)*(abs(V_max-V_min))/zrange+V_min;

    % give some random variation in signal amplitude
    s_clean = s_clean*rand_uniform_centered(1,0.1,1);
    
    % time for nirs and rr
    time = 0:1/fs:length(s_clean)/fs-1/fs;
    time_rrn = Tspan;
    time_rr = t_rrup;
    hr_inst = 1./rr;
    
    %% final time segment
    % Return a 60 s window [t0, t1] with hardcoded indices below.
    % Adjust these if you want different segment durations.
    t0 = 10; t1 = 70; % second
    t_seg = time(t0*fs+1:t1*fs)';
    t_seg = t_seg-t_seg(1);
    s_seg = s_clean(t0*fs+1:t1*fs);

    %% to visualize
    if plt_fig == 1
        % plot full signal, spectrogram, and RR Interval
        color_all = {'#9C62F5','#405FC1','#EB5055','#FFAA1C','#62A140'};
    
        figure; set(gcf,'Position',[0,100,800,700]);
        % rr interval and hr
        subplot(3,1,3)
        yyaxis left
        plot(time_rrn,rrn,'-','Color',color_all{2}); hold on
        plot(time_rr+rr(1)/2,rr,'-','Color',color_all{4});
        ylabel('Time (s)','Color','k');
        xlabel('Time (s)');
        title('(c) RR Interval Time Trace');
        box off
        
        yyaxis right
        plot(time_rr+rr(1)/2,60*hr_inst,'Color',color_all{3});
        ylabel('Instantaneous HR (bpm)','Color',color_all{3});
        legend('rrn','rr','HR'); legend AutoUpdate off
        xlim([t0-5 t1+5]);
        xline([t0 t1],'k--','LineWidth',1.5);
        
        % spectrogram
        seg_len = floor(length(time)*0.075);
        overlap_percent = 0.9;
        overlap = floor(overlap_percent*seg_len);
        NFFT = floor(2^nextpow2(length(time))*0.75);
        
        h = subplot(3,1,2);
        I_plt = s_clean;
        spectrogram(I_plt,seg_len,overlap,NFFT,fs,'yaxis','power');
        title('(b) Spectrogram of (a)');
        ylim([0 5]);
        xlim([(t0-5)/60 (t1+5)/60]);
        h.XTickLabel = string(h.XTick*60);
        xlabel('Time (s)');
        colorbar off
        caxis('auto');
        xline([t0/60 t1/60],'k--','LineWidth',1.5);
        
        % dOD
        subplot(3,1,1)
        plot(time,s_clean,'Color',color_all{1});
        title('(a) Synthetic Signal');
        ylabel('\DeltaOD (a.u.)');
        xlabel('Tims (s)');
        xlim([t0-5 t1+5]);
        xline([t0 t1],'k--','LineWidth',1.5);
        ylim([-1.5 1.5]);
        box off
    
        figure; set(gcf,'position',[0,100,600,300]);
        plot(t_seg,s_seg,'color',color_all{1});
        title('Zoomed in Signal');
        ylabel('\DeltaOD (a.u.)');
        xlabel('Time (s)');
        xlim([5 10]);
        box off
    
        [freq,amp] = fft_basic(s_seg,fs,1,0);
        figure; set(gcf,'Position',[600,100,400,400]);
        plot(freq,amp);
        title('FFT of Signal');
        ylabel('Amplitude (a.u.)');
        xlabel('Frequency (Hz)');
        box off
        xlim([0.05 5]);
    end
end

