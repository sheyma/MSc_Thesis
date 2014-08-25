
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
dlmwrite('freq_matlab.dat', fre, 'delimiter','\t', 'precision', '%.6f');

Wn    = 0.01;   % Hz , cut-off freq 
[Bs,As] = butter(5,Wn,'low');

X_filt     = filtfilt(Bs,As,X);
X_filt_fft = fft(X_filt, N_pow) /N;
X_filt_fft = 2*abs(X_filt_fft(1:N_pow /2 +1)) ; 

x_python = load('sinwave.dat');
t_python = load('sinwave_time.dat');

xf_python = load('sinwave_fft.dat');
tf_python = load('sinwave_fre.dat');

x_filt_python = load('sinwave_filt.dat');
x_filt_fft_py = load('sinwave_filt_fre.dat');

%%%%
y_python = load('bold_y.dat');
yt_python= load('bold_yt.dat');

y_fft_py = load('bold_y_fft.dat');
y_fre_py = load('bold_yfre.dat');

y_filt_py = load('bold_y_filt.dat');
y_filt_ff = load('bold_y_filt_fft.dat');


figure(1);
subplot(1,2,1)
hold on
plot(t,x+noise, 'b')

plot(t_python , x_python , '.r')
hold off
ylabel('signal')
xlabel('time [s]')
legend('matlab' , 'python')
subplot(1,2,2)
hold on
plot(fre(1:100) ,Xfft(1:100) )
plot(tf_python(1:100) , xf_python(1:100), 'r'  )
legend('matlab' , 'python')
ylabel('|signal(f)|')
xlabel('f [Hz]')

figure(2)
subplot(1,2,1)
hold on
plot(t, X_filt, 'b')
plot(t_python , x_filt_python, '.r', 'LineWidth',1)
hold off
legend('matlab' , 'python')
ylabel('signal filt')
xlabel('time [s]')
axis([0 , 100, -2 , 2])
subplot(1,2,2)
hold on
plot(fre(1:100), X_filt_fft(1:100))
plot(tf_python(1:100) , x_filt_fft_py(1:100), 'r')
hold off
legend('matlab' , 'python')
ylabel('|signal filt (f)|')
xlabel('f [Hz]')



f_c       = 0.25;
dtt       = 0.001;  % (here 1 millisecond)
f_s       = 1/dtt;  % Sampling frequency (Hz)
f_N       = f_s/2;  % Nyquist frequency (Hz)

y     = load('bold_signal_matlab.dat');
y     = y(:,1);
t_y   = 0:length(y)-1;

M     = length(y);
M_pow = 2^nextpow2(M);
yfft  = fft(y , M_pow) /M;
yfft  = 2*abs(yfft(1:M_pow /2 +1)) ; 
freq  = f_s/2 * linspace(0,1, M_pow/2 + 1);



[Bs,As] = butter(5,f_c/f_N,'low');
y_filt  = filtfilt(Bs,As,y);
y_filt_fft  = fft(y_filt , M_pow) /M;
y_filt_fft  = 2*abs(y_filt_fft(1:M_pow /2 +1)) ; 


figure(3);
subplot(1,2,1)
hold on 
plot(t_y /1000  , y, 'b')
plot(yt_python/1000 , y_python , '*r', 'MarkerSize', 1);
hold off
legend('matlab' , 'python')
xlabel('time [s]')
ylabel('bold signal')
%axis([0, t(end), 4 , 5.5])
subplot(1,2,2)
hold on
plot(freq(1:50), yfft(1:50))
plot(y_fre_py(1:50) , y_fft_py(1:50), 'r')
hold off
legend('matlab' , 'python')
ylabel('|bold signal (f)|')
xlabel('f [Hz]')


figure(4)
subplot(1,2,1)
hold on 
plot(t_y/1000, y_filt, 'b')
plot(yt_python/1000, y_filt_py, 'r' )
xlabel('time [s]')
ylabel('bold signal filt')
legend('matlab' , 'python')
%axis([0, t(end), 4.9 , 5.4])
subplot(1,2,2)
hold on 
plot(freq(1:50) , y_filt_fft(1:50), 'b')
plot(y_fre_py(1:50) , y_filt_ff(1:50), 'r')
legend('matlab' , 'python')
ylabel('|bold signal filt (f)|')
xlabel('f [Hz]')


