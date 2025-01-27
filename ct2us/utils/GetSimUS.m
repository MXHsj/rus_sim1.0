%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% file name:    GetSimUS.m
% author:       Xihan Ma
% description:  simulate ultrasound image given scattering image
% date:         2024-03-04
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function [xi, zi, I] = GetSimUS(probe, x, y, z, RC, num_series, tiltAng, isVis)

if nargin < 8
    isVis = false;
end

tilt = deg2rad(linspace(-tiltAng,tiltAng,num_series)); % tilt angles in rad

txdel = cell(num_series,1); % this cell will contain the transmit delays
for k = 1:num_series
%     txdel{k} = txdelay(param,tilt(k),deg2rad(60));
    txdel{k} = txdelay(probe,tilt(k));
end
tic
RF = cell(num_series,1); % this cell will contain the RF series
probe.fs = 4*probe.fc; % sampling frequency in Hz

% Simulate the seven series of RF signals with SIMUS. The RF signals will be sampled at 4 $\times$ center frequency.
option.WaitBar = false; % remove the wait bar of SIMUS
option.ParPool = false; % parallel processing
h = waitbar(0,'');
for k = 1:num_series
    waitbar(k/num_series,h,['SIMUS: RF series #' int2str(k) ' of ', num2str(num_series)])
    RF{k} = simus(x,y,z,RC,txdel{k},probe,option);
end
close(h)
fprintf('simulate RF signal took %.3f [sec]\n', toc)

tic
IQ = cell(num_series,1);  % this cell will contain the I/Q series
for k = 1:num_series
    IQ{k} = rf2iq(RF{k},probe.fs,probe.fc);
end

[xi,zi] = impolgrid([256 128],15e-2,deg2rad(80),probe);
bIQ = zeros(256,128,num_series);  % this array will contain the 7 I/Q images
h = waitbar(0,'');
fprintf('simulate IQ data took %.3f [sec]\n', toc)

tic
for k = 1:num_series
    waitbar(k/7,h,['DAS: I/Q series #' int2str(k) ' of ', num2str(num_series)])
    bIQ(:,:,k) = das(IQ{k},xi,zi,txdel{k},probe);
end
close(h)
bIQ = tgc(bIQ);
fprintf('beamform took %.3f [sec]\n', toc)

cIQ = sum(bIQ,3); % this is the compound beamformed I/Q
I = bmode(cIQ,50); % log-compressed image

%% vis
if isVis
    ShowUS(xi, zi, I)
end
