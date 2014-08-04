% calculate BOLD signal from pydelay/netpy output
% ###############################################

function b = calcBOLD(simfile)

	%% simfile should be the output file from the python script

  disp(simfile ) 

  tic;
  simoutput = dlmread(simfile);
  toc;

	%% get number of network nodes N and read u_i timeseries from simfile:
  tvec = simoutput(:,1);

  nt = size(tvec,1)
  dt = tvec(2)-tvec(1)

  N = (size(simoutput,2)-1)/2

  timeseries = zeros(size(simoutput,1),N);
  size(timeseries)

  for roi = 1:N
    timeseries(:,roi) = simoutput(:,2*roi);
  end
  dlmwrite('bold_timeseries_matlab.dat', timeseries, 'delimiter','\t', 'precision', '%.6f');

  %save([simfile(1:end-4),'_timeseries.mat'],'timeseries','tvec')
  %load([simfile(1:end-4),'_timeseries.mat'])
  
  %% plot sample time series     
  
  % specify plotting interval: 
  minval = 325000;
  range = 500;
  h = figure;
  plot(timeseries(minval:minval+range,:));
  xlim([0 range])
  xlabel('t in [ms]')
  ylabel('u(t)')
  
  textobj = findobj('type', 'text');
  set(textobj, 'fontunits', 'points');
  set(textobj, 'fontsize', 60);
  
  %filo = ['sample_',simfile(1:end-4)]; 
  %print(h,'-depsc2',sprintf('%s.eps',filo));
  %system(sprintf('ps2pdf -dEPSCrop %s.eps %s.pdf',filo,filo));  
  close(h);
  
  %%% apply Balloon Windkessel model in BOLD.m :  

  % initialize array:
  boldsignal = cell(N,1);
	
	% important: specify here to which time interval the simulated 
	% time series corresponds:
  dt_BOLD = 0.001 ;
  T = round(tvec(end) /dt * dt_BOLD) 
   
%T =450.0; % in [s]
  
var = [] ; 
  for roi = 1:N 
    boldsignal{roi} = BOLD(T,timeseries(:,roi));
    var = [var, boldsignal{roi}];
    disp(roi)
    % verify that there is no errors in the BOLD results
    nans = size(find(isnan(boldsignal{roi})),1);
    if nans > 0
      disp(nans)
    end
  end
  
  
  dlmwrite('bold_signal_matlab.dat', var, 'delimiter','\t', 'precision', '%.6f');
    
  
  %% filter below 0.25Hz:

  f_c=0.25;
  dtt=0.001; % Resolution of the BOLD signal (here 1 millisecond).

  % Low-Pass filter the BOLD signal

  n_t       = size(boldsignal{1},1)
  BOLD_filt = zeros(n_t,N);
  f_s       = 1/dtt                  % Sampling frequency (Hz)
  f_N       = f_s/2                 % Nyquist frequency (Hz)

  % Calculate variables for Butterworth lowpass filter of order 5 
  % with cut off frequency f_c/f_N
  [Bs,As] = butter(5,f_c/f_N,'low')
  dlmwrite('Bs_matlab.dat', Bs, 'delimiter','\t', 'precision', '%.25f'); 
  dlmwrite('As_matlab.dat', As, 'delimiter','\t', 'precision', '%.25f');
  
  size(BOLD_filt)

  for n = 1:N
    x               = boldsignal{n};
    BOLD_filt(:,n)  = filtfilt(Bs,As,x); % Apply filter
    plot(BOLD_filt)
    %size(BOLD_filt)
  end
  dlmwrite('bold_filt_matlab.dat', BOLD_filt, 'delimiter','\t', 'precision', '%.6f');  

  %% Downsampling: select one point every 'ds' ms to match fmri resolution:
  %BOLD_filt = load('bold_filt_matlab.dat');  
  ds=2.500; 
  down_bds=BOLD_filt(1:ds/dtt:end,:);
  lenBold = size(down_bds,1);
  dlmwrite('bold_down_matlab.dat', down_bds, 'delimiter','\t', 'precision', '%.6f');
  
  %% Cutting first and last seconds (distorted from filtering) and keep the middle:
  cut_percent  = 2/100;
  limit_down   = ceil(lenBold * cut_percent)
  limit_up     = ceil(lenBold - limit_down-2)
  index        = limit_down : limit_up 
  bds          = down_bds(index, :);
  %nFramesToKeep = 260; % use 260!!
  %bds = down_bds(floor((lenBold-nFramesToKeep)/2):floor((lenBold+nFramesToKeep)/2)-1,:);
  
  size(bds)  
  dlmwrite('bold_cut_matlab.dat', bds, 'delimiter','\t', 'precision', '%.6f');
  %%
  
  %load([simfile(1:end-4),'_bds.mat'])

  simfc = corr(bds);
  %save([simfile(1:end-4),'_simfc.mat'],'simfc')
  dlmwrite('bold_corr_matlab.dat', simfc, 'delimiter','\t', 'precision', '%.10f');

  
   % plot simulated functional connectivity
  h = figure;
  imagesc(simfc); % automatic color scaling from min to max value 
  %imagesc(simfc,[-1.0 1.0]); chose this for color scaling from -1 to 1
  colorbar;

  textobj = findobj('type', 'text');
  set(textobj, 'fontunits', 'points');
  set(textobj, 'fontsize', 60);

  filo = ['simfc_',simfile(1:end-4)]; 
  print(h,'-depsc2',sprintf('%s.eps',filo));
  system(sprintf('ps2pdf -dEPSCrop %s.eps %s.pdf',filo,filo));
  
end


