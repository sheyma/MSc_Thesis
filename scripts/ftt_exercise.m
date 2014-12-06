% exporting data needed for (ftt_exercise...).py !!!

v1 = 0.05;   %Hz
v2 = 0.10;  %Hz

Fs = 10;    % Hz , Sampling frequency
T = 1/Fs;   % s, Sample time
tmax = 100; % s, max time of signal run

t = 0: T : tmax -T ; % time array 
dlmwrite('t_matlab.dat', t, 'delimiter','\t', 'precision', '%.6f');

x = sin(2*pi*v1*t) + sin(2*pi*v2*t); 

%noise = randn(size(t));
noise = 0;
X     = x+noise;
dlmwrite('sine_matlab.dat', X, 'delimiter','\t', 'precision', '%.6f');

N     = length(x);
N_pow = 2^nextpow2(N);

Xfft  = fft(X , N_pow) /N;
Xfft  = 2*abs(Xfft(1:N_pow /2 +1)) ; 
dlmwrite('sine_fft_matlab.dat', Xfft, 'delimiter','\t', 'precision', '%.6f');

fre   = Fs/2 * linspace(0,1, N_pow/2 + 1);
dlmwrite('sin_fre_matlab.dat', fre, 'delimiter','\t', 'precision', '%.6f');

Wn    = 0.01;   % Hz , cut-off freq 
[Bs,As] = butter(5,Wn,'low');

X_filt     = filtfilt(Bs,As,X);
X_filt_fft = fft(X_filt, N_pow) /N;
X_filt_fft = 2*abs(X_filt_fft(1:N_pow /2 +1)) ; 

dlmwrite('X_filt_matlab.dat', X_filt, 'delimiter','\t', 'precision', '%.6f');
dlmwrite('X_filt_fft_matlab.dat', X_filt_fft, 'delimiter','\t', 'precision', '%.6f');

f_c       = 0.25;
dtt       = 0.001;  % (here 1 millisecond)
f_s       = 1/dtt;  % Sampling frequency (Hz)
f_N       = f_s/2;  % Nyquist frequency (Hz)

y     = load('bold_signal_matlab.dat');
y     = y(:,1);
t_y   = 0:length(y)-1;
dlmwrite('t_y_matlab.dat', t_y, 'delimiter','\t', 'precision', '%.6f');
dlmwrite('y_matlab.dat', y, 'delimiter','\t', 'precision', '%.6f');

% LOAD a sample FHN time series, extract 1st column 
%A_ =load('A_aal_0_ADJ_thr_0.60_sigma=0.2_D=0.05_v=70.0_tmax=45000.dat');
A_ =load('acp_w_0_ADJ_thr_0.50_sigma=0.3_D=0.05_v=60.0_tmax=45000.dat');
A  = A_(:,2:91);

% LOAD a sample BOLD time series, there is no time column in it
%A = load('A_aal_0_ADJ_thr_0.66_sigma=0.03_D=0.05_v=70.0_tmax=45000_NORM_BOLD_signal.dat');
A = load('acp_w_0_ADJ_thr_0.54_sigma=0.03_D=0.05_v=30.0_tmax=45000_NORM_BOLD_signal.dat');

M     = length(A(:,1));
M_pow = 2^nextpow2(M);
freq  = f_s/2 * linspace(0,1, M_pow/2 + 1);
YFFT  = [];
%iter = 25000 ; % FHN
iter = 1000; %BOLD
N = [];
F = [];
k = 1; 

f  = freq(1:iter);
for i=1:90;
    
    yfft  = fft(A(:,i) , M_pow) /M;
    yfft  = 2*abs(yfft(1:M_pow /2 +1)) ; 
    yfft  = yfft(1:iter);   
    YFFT  = [YFFT, yfft'] ;
     
    for j=1:length(yfft);
        N(k) = i;
        F(k) = f(j);
        k = k+1;
    end
    yfft=[];
end

FFT = log(YFFT); 

[xq, yq] = meshgrid(min(N):0.01:max(N),min(F):0.01:max(F));
vq = griddata(N,F,FFT, xq, yq);

figure(3) ; mesh(xq, yq, vq); colorbar
%set(gca,'ZScale','log');
zlabel('$\log \hat{f}(\nu)$', 'Interpreter', 'Latex', 'fontsize',30)
xlabel('Nodes', 'fontsize' , 25)
ylabel('$\nu$[Hz]', 'Interpreter', 'Latex', 'fontsize',25)
set(gca, 'fontsize', 25)

figure(4) ; surf(xq,yq,vq);


    
%dlmwrite('freq_matlab.dat', freq, 'delimiter','\t', 'precision', '%.6f');
%dlmwrite('yfft_matlab.dat', yfft, 'delimiter','\t', 'precision', '%.6f');






[Bs,As] = butter(5,f_c/f_N,'low');
y_filt  = filtfilt(Bs,As,y);
y_filt_fft  = fft(y_filt , M_pow) /M;
y_filt_fft  = 2*abs(y_filt_fft(1:M_pow /2 +1)) ; 
dlmwrite('y_filt_matlab.dat', y_filt, 'delimiter','\t', 'precision', '%.6f');
dlmwrite('y_filt_fft_matlab.dat', y_filt_fft, 'delimiter','\t', 'precision', '%.6f');