%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% file name:    GetScatterImage.m
% author:       Xihan Ma
% description:  generate scattering image given a scatter map
% date:         2024-03-04
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function [x,y,z,RC] = GetScatterImage(probe, depth, scatter_map, isVis)

if nargin < 4
    isVis = false;
end
if ~isa(scatter_map, 'single')
    scatter_map = single(scatter_map);
end

speedOfSound = 1540;

% probe.TXapodization = cos(linspace(-1,1,probe.Nelements)*pi/2);

[x,y,z,RC] = genscat([NaN, depth*1e-3], speedOfSound/probe.fc, scatter_map);

%% vis
if isVis
    scatter(x*1e2,z*1e2,2,20*log10(RC/max(RC(:))),'filled')
    caxis([-40 0])
    colormap hot
    axis equal ij tight
    set(gca,'XColor','none','box','off')
    title([int2str(numel(RC)) ' tissue scatterers'])
    ylabel('[cm]')
end
