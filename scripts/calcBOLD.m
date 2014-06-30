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
  T =450.0; % in [s]
  
  for roi = 1:N 
    boldsignal{roi} = BOLD(T,timeseries(:,roi));
    disp(roi)
    % verify that there is no errors in the BOLD results
    nans = size(find(isnan(boldsignal{roi})),1);
    if nans > 0
      disp(nans)
    end
  end
  
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

  size(BOLD_filt)

  for n = 1:N
    x               = boldsignal{n};
    BOLD_filt(:,n)  = filtfilt(Bs,As,x) % Apply filter
    plot(BOLD_filt)
    %size(BOLD_filt)
  end


  %% Downsampling: select one point every 'ds' ms to match fmri resolution:

  ds=2.500; 
  down_bds=BOLD_filt(1:ds/dtt:end,:);
  lenBold = size(down_bds,1);
  
  %% Cutting first and last seconds (distorted from filtering) and keep the middle:
  nFramesToKeep = 260;
  bds = down_bds(floor((lenBold-nFramesToKeep)/2):floor((lenBold+nFramesToKeep)/2)-1,:);
  size(bds)  
  save([simfile(1:end-4),'_bds.mat'],'bds')

  %%
  
  %load([simfile(1:end-4),'_bds.mat'])

  simfc = corr(bds);
  save([simfile(1:end-4),'_simfc.mat'],'simfc')
  
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


