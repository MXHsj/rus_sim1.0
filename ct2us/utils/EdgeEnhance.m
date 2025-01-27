%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% file name:    EdgeEnhance.m
% author:       Xihan Ma
% description:  enhance CT slice by highlighting tissue boundary
% date:         2024-02-29
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function [out] = EdgeEnhance(in, isVis)

if nargin < 2
    isVis = false;
end
if ~isa(in, 'single')
    in = single(in);
end

if max(in(:)) > 1 || min(in(:) < 0)
    disp('edge enhance: input image is not normalized')
    in = rescale(in, 0, 1);
end

% ========== params ==========
CANNY_THRESH = [0.05, 0.2];     % [0.08, 0.2]
CANNY_SIGMA = 4.0;
% ============================

edge_map = edge(in, 'Canny', CANNY_THRESH, CANNY_SIGMA);
out = in;
out(edge_map) = 1;

%% visualize results
if isVis
    display_height = size(in, 1);
    display_width = size(in, 2);
    figure('Position', 0.8*[0, 0, round(2*display_width), round(display_height)]);
    tiledlayout(1,2,'TileSpacing','none')
    ax1 = nexttile;
    imagesc(out); colormap(ax1, 'gray'); axis image off;
    ax2 = nexttile;
    imagesc(edge_map); colormap(ax2, 'gray'); axis image off;
end


