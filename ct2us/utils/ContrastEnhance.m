%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% file name:    ContrastEnhance.m
% author:       Xihan Ma
% description:  enhance CT slice contrast, returns enhanced image and
%               dynamic range for display
% date:         2024-02-29
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function [out, out_clampped, drange] = ContrastEnhance(in, isVis)
% :param in:            input 2D B-mode OCT image
% :param isVis:         flag to enable result visualization
% :return out:          contrast enhanced image
% :return out_clamped:  contrast enhanced image with optimal drange applied
% :return drange_out:   optimal dynamic range

if nargin < 2
    isVis = false;
end
if ~isa(in, 'single')
    in = single(in);
end

if max(in(:)) > 1 || min(in(:) < 0)
    disp('contrast enhance: input image is not normalized')
    in = rescale(in, 0, 1);
end

% ===== apply optimal drange on input image
% drange_in = 0.9*[mean(in,'all'), max(in,[],'all')];
in_clampped = in;
% in_clampped(in_clampped > drange_in(2)) = drange_in(2);
% in_clampped = in_clampped - drange_in(1);

% ===== contrast enhancement
height = size(in, 1); width = size(in, 2);
num_tiles = round(0.02*[height, width]);
distr = 'uniform';
out = adapthisteq(in_clampped, 'NumTiles', num_tiles, ...
                  'Distribution', distr, ...
                  'Range','original');

% out = adapthisteq(in_clampped);

% ===== brightness enhancement
out = imadjust(out);
                  
% ===== apply optimal drange on output image
% drange = 0.9*[mean(out,'all'), max(out,[],'all')];
out_clampped = out;
% out_clampped(out_clampped > drange(2)) = drange(2);
% out_clampped = out_clampped - drange(1);

%% visualize results
if isVis
    display_height = size(in, 1);
    display_width = size(in, 2);
    figure('Position', 0.8*[0, 0, round(2*display_width), round(display_height)]);
    tiledlayout(1,2,'TileSpacing','none')
    ax1 = nexttile;
    imagesc(in); colormap(ax1, 'gray'); axis image off;
    ax2 = nexttile;
    imagesc(out_clampped); colormap(ax2, 'gray'); axis image off;
%     imshow([in, out])
end