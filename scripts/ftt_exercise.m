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

M     = length(y)
M_pow = 2^nextpow2(M)
yfft  = fft(y , M_pow) /M;
yfft  = 2*abs(yfft(1:M_pow /2 +1)) ; 
freq  = f_s/2 * linspace(0,1, M_pow/2 + 1);
dlmwrite('freq_matlab.dat', freq, 'delimiter','\t', 'precision', '%.6f');
dlmwrite('yfft_matlab.dat', yfft, 'delimiter','\t', 'precision', '%.6f');

[Bs,As] = butter(5,f_c/f_N,'low');
y_filt  = filtfilt(Bs,As,y);
y_filt_fft  = fft(y_filt , M_pow) /M;
y_filt_fft  = 2*abs(y_filt_fft(1:M_pow /2 +1)) ; 
dlmwrite('y_filt_matlab.dat', y_filt, 'delimiter','\t', 'precision', '%.6f');
dlmwrite('y_filt_fft_matlab.dat', y_filt_fft, 'delimiter','\t', 'precision', '%.6f');