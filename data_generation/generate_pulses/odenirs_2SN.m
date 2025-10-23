function dxdt = odenirs_2SN(t,x,flag,rr,sfint,ai,location,scale_hr,shape,f_resp,amp_resp,drift_resp)
    % This function was modified from the derivsecgsyn function from ECGSYN: https://physionet.org/content/ecgsyn/1.0.0/
    % It was modifed for our work to take in the 2 skewed normal waveforms.
    % rr: rr process 
    % sfint: Internal sampling frequency (Hz)
    % ai, location, scale_hr,shape: parameters for the skewed gaussians
    % f_resp: respiration frequency (Hz)
    % amp_resp: respiration amplitude
    % drift_resp: 1-include repiration drift, 0-not

    % This funcion descripes the right-hand side of a 3-state oscillator 
    % driven by an RR process, adapted from ECGSYN’s derivsecgsyn. 
    % In this work, the third state accumulates a pair of skewed-normal 
    % waveform derivatives to synthesize a pulsatile signal with 
    % controllable morphology (two components).
    %
    % State vector
    %   x(1), x(2) : phase-plane oscillator states (unit-radius attractor)
    %   x(3)       : pulsatile amplitude channel (clean signal prior to scaling)
    %
    % Inputs
    %   t        : time (s)
    %   x        : state vector [x1; x2; x3]
    %   flag     : (unused; kept for ODE signature compatibility)
    %   rr       : RR interval/process
    %   sfint    : internal sampling frequency (Hz)
    %   ai       : [a1 a2], amplitude weights for the two skew-normal lobes
    %   location : [θ1 θ2], lobe centers (radians)
    %   scale_hr : per-time scales for each lobe, size length(rr) × 2
    %   shape    : [κ1 κ2], skewness parameters for each lobe
    %   f_resp   : respiration frequency (Hz)
    %   amp_resp : respiration baseline amplitude (a.u.)
    %   drift_resp : 1 to include sinusoidal baseline drift, 0 to omit
    %
    % Output
    %   dxdt     : time derivative [dx1/dt; dx2/dt; dx3/dt]
    %
    % Author: Jingyi Wu, 2025

    % SPDX-License-Identifier: GPL-3.0-or-later
    
    % Parameters for the limit cycle
    ta = atan2(x(2),x(1)); % phase angle
    r0 = 1; % radial contraction toward unit circle
    a0 = 1.0 - sqrt(x(1)^2 + x(2)^2)/r0;
    ip = 1+floor(t*sfint); % index into RR sequence at internal rate
    w0 = 2*pi/rr(ip); % instantaneous angular speed from RR
    
    % Optional respiration baseline drift in x(3)
    if drift_resp == 1
        zbase = amp_resp*sin(2*pi*f_resp*t);
    elseif drift_resp == 0
        zbase = zeros(length(t),1);
    end

    % Phase-plane oscillator (unit-radius stable limit cycle)
    dx1dt = a0*x(1) - w0*x(2);
    dx2dt = a0*x(2) + w0*x(1);
    
    dti = rem(ta - location, 2*pi);
    x0 = dti;
    scale = scale_hr(ip,:);
    
    % Two skew-normal derivative lobes (systolic/diastolic components)
    dx3dt_g1 = ai(1)*dsndx(x0(1),location(1),scale(1),shape(1));
    dx3dt_g2 = ai(2)*dsndx(x0(2),location(2),scale(2),shape(2));

    dx3dt = dx3dt_g1+dx3dt_g2- 1.0*(x(3) - zbase);

    dxdt = [dx1dt; dx2dt; dx3dt];

end
