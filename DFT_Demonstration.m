clear all
close all

%% Example signal - Sinusoidal w/ 3 frequencies, N samples
pi = 3.14;
SR = 100;
f1 = 1;
f2 = 6;
f3 = 16;
Tdur = 10;

t = [0: 1/SR: Tdur - (1/SR)];
xsig = sin(2*pi*f1*t) + sin(2*pi*f2*t).*(t>Tdur*0.5) + sin(2*pi*f3*t).*(t>Tdur*0.8);
fig = figure(1); plot(t,xsig);
fig.Position = [0 480 450 300];
title('Time Signal');
N = size(xsig,2);

%% FFT parameters Tdur * 
n_fft = 20;

%% DFT Calculation for whole signal
for k= 1:n_fft
    X(k) = 0;
    for n = 1:N
        f_sample(k) = Tdur * (k/N); %#ok<*SAGROW>
        X(k) = X(k) + (xsig(n)* exp(-i*2*pi  *  n  *  f_sample(k) ) );
    end
end
X=X'; f = f_sample*SR;
X_abs(:) = abs(X(:));
fig = figure(2); plot(f_sample*SR , X_abs, 'o'); % From 1/samples to 1/seconds
fig.Position = [450 480 450 300];
title('DFT for whole signal');

%% Split xsig into segments
nSeg=5;
xsig_seg = reshape(xsig,(1/nSeg) * SR * Tdur,[])';
Nseg = SR * (Tdur/nSeg);

fig = figure(12);
fig.Position = [450 480 800 300];
subplot(2,nSeg,[1:nSeg]);
plot(xsig);
title(['Sinal Original  - S_o_r_i_g_i_n_a_l '])

arrSegs = 1 + [1:nSeg];
for idx = 1:nSeg
   subplot(2,nSeg,nSeg+idx) ;
   plot(xsig_seg(idx,:));
   axis([0 size(xsig,2)/nSeg -4 4])
   title(['S para T = ' num2str(idx)])
end


%% DFT Calculation for segments, w/ heatmap
for s=1:nSeg
    for k=1:n_fft
        Xseg(k,s) = 0;
        for n = 1:Nseg
            fseg_sample(k) = Tdur*(k/N);
            Xseg(k,s) = Xseg(k,s) + (xsig_seg(s,n)* exp(-1i*2*pi  *  n  *  fseg_sample(k) ) );
        end
    end
end
fseg = fseg_sample*SR;
Xseg_abs = abs(Xseg);
fig = figure(3);
fig.Position = [900 480 450 300];
hm = heatmap(Xseg_abs); 
for i=1:n_fft
   hm.YData{i} = fseg(i); 
end
title('Spectogram from DFT');

%% Hanning window
for n = 1: Nseg
  w(n) = 0.5 - 0.5 * cos(2*pi*n/Nseg);
end
fig = figure(4); plot(w);
fig.Position = [0 100 450 300];
title('Janela Hanning ');
N = size(xsig,2);

% Apply windowing to all segments
xsig_seg_win(:,:) = xsig_seg(:,:) .* w(:,:);

% Plot one example
fig = figure(5);
subplot(2,1,1);
plot(xsig_seg(1,:));
title('Antes do janelamento');
subplot(2,1,2);
plot(xsig_seg_win(1,:));
title('Após o janelamento aplicando janela Hanning');
fig.Position = [450 100 450 300];

%% Zero padding
N_sw = size(xsig_seg_win,2)

for idx = 1:30
    M = 2^idx;
    if M > N_sw
        break;
    end
end
M=N_sw;

I = eye(N_sw);
O = zeros( N_sw , M-N_sw);

xsig_seg_win_zp(:,:) = xsig_seg_win(:,:) * [I O]; 

% Plot one example
fig = figure(6);
subplot(2,1,1);
plot(xsig_seg_win(1,:));
title('Antes do zero-padding');
subplot(2,1,2);
plot(xsig_seg_win_zp(1,:));
axis([0 M -1 1])
title('Após o zero-padding');
fig.Position = [900 100 450 300];

%% Discrete fourier coefficients
W = dftmtx(M);
W = W(1:M, 1:M/2);

FDC(:,:) = xsig_seg_win_zp(:,:) * W;

%% Power spectrum calculation
FDC_power(:,:) = FDC(:,:) .* conj(FDC(:,:));

Xseg_abs = abs(Xseg);
fig = figure(7);
fig.Position = [0 480 450 300];

for idx=1:size(FDC_power,2)%n_fft
   XData_tmp(idx) = (idx-1) / (Nseg / SR) ; 
end

hm = heatmap(flip(FDC_power(:, and(floor(XData_tmp) == XData_tmp , XData_tmp < f3+1)))'); 
XData_tmp = XData_tmp(and(floor(XData_tmp) == XData_tmp , XData_tmp < f3+1));

for idx=1:size(XData_tmp,2)
     hm.YData{idx} = XData_tmp(idx) ; 
end
title('Cálculo de energia espectral / Espectrograma');
ylabel('Frequências [Hz]');
xlabel('Número do Segmento [-]');

%% Mel filters calculation and application
nFilters = 25
melFil = melfilter(nFilters, [1:1:M/2])';
fig = figure(8);
fig.Position = [450 480 450 300];
plot(melFil);
title('Mel Filters');

logFilFC_power = log10( FDC_power(:,:) * melFil(:,:) );

fig = figure(9);
fig.Position = [900 480 450 300];
hm = heatmap(logFilFC_power(:,:)); 
for i=1:nFilters
   hm.XData{i} = i-1; 
end
title('Log Filtered Powers');
xlabel('Frequencies');
ylabel('Time elapsed');

%% Discrete cosine transform

MFCC = logFilFC_power(:,:) * dctmtx(nFilters);
fig = figure(10);
fig.Position = [900 100 450 300];
hm = heatmap(MFCC(:,:)); 


%% Final comparison with base function

%MFCC2 = mfcc(xsig, SR)
