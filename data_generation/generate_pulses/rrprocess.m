function x = rrprocess(flo, fhi, flostd, fhistd, lfhfratio, sfrr, n)
    % This function computes the rr process.
    % From ECGSYN: https://physionet.org/content/ecgsyn/1.0.0/
    w1 = 2*pi*flo;
    w2 = 2*pi*fhi;
    c1 = 2*pi*flostd;
    c2 = 2*pi*fhistd;
    sig2 = 1;
    sig1 = lfhfratio;
    
    df = sfrr/n;
    w = [0:n-1]'*2*pi*df;
    dw1 = w-w1;
    dw2 = w-w2;
    
    Hw1 = sig1*exp(-0.5*(dw1/c1).^2)/sqrt(2*pi*c1^2);
    Hw2 = sig2*exp(-0.5*(dw2/c2).^2)/sqrt(2*pi*c2^2);
    Hw = Hw1 + Hw2;
    Hw0 = [Hw(1:n/2); Hw(n/2:-1:1)];
    Sw = (sfrr/2)*sqrt(Hw0);
    
    ph0 = 2*pi*rand(n/2-1,1);
    ph = [ 0; ph0; 0; -flipud(ph0) ]; 
    SwC = Sw .* exp(1i*ph);
    x = (1/n)*real(ifft(SwC));
    
end