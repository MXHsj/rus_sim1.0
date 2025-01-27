%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% file name:    ShowUS.m
% author:       Xihan Ma
% description:  show simulated ultrasound image obtained from MUST
% date:         2024-03-04
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function ShowUS(xi, zi, I, isMarkUp)

if nargin < 4
    isMarkUp = false;
end

pcolor(xi*1e2,zi*1e2,I)
shading interp, colormap gray

if isMarkUp
    axis equal ij
    set(gca,'XColor','none','box','off')
    c = colorbar;
    c.YTick = [0 255];
    c.YTickLabel = {'-50 dB','0 dB'};
    ylabel('[cm]')
else
    axis equal ij off
end

