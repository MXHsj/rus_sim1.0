%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% file name:    resize.m
% author:       Xihan Ma
% description:  adjust image to desired size but keep ratio by cropping
% date:         2024-03-08
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = resize(in, size_d, axis)
% NOTE: the output image might lose information due to cropping
    if nargin < 3
        axis = 1;   % axis to adjust ratio by cropping/padding
    end
    
    assert(axis == 1 || axis == 2, 'invalid input, axis must either be axial (1) or lateral (2)')
    assert(length(size_d) == 2, 'incorrect input for desired size: [height, width]')
    
    max_val = double(max(in, [], 'all'));
    dim = length(size(in));
    height_d = size_d(1);
    width_d = size_d(2);
    ratio_d = width_d/height_d;
    
    if axis == 1

        resized = imresize(in, width_d/size(in, 2));
        diff = abs(size(resized, 1) - height_d);
        
        if size(resized, 1) > height_d
            % will vertically crop input
            if dim == 3
                out = resized(round(diff/2):height_d+round(diff/2)-1, :, :);
            elseif dim == 2
                out = resized(round(diff/2):height_d+round(diff/2)-1, :);
            end
        else
            % will vertically pad input
            if dim == 3
                out = [max_val*ones(round(diff/2), size(resized, 2), 3); ...
                       resized; ...
                       max_val*ones(round(diff/2), size(resized, 2), 3)];
            elseif dim == 2
                out = [max_val*ones(round(diff/2), size(resized, 2)); ...
                       resized; ...
                       max_val*ones(round(diff/2), size(resized, 2))];
            end
        end

    elseif axis == 2
        
        resized = imresize(in, height_d/size(in, 1));
        diff = abs(size(resized, 2) - width_d);

        if size(resized, 2) > width_d
            % will horizontally crop input
            diff = abs(size(resized, 2) - width_d);
            if dim == 3
                out = resized(:, round(diff/2):width_d+round(diff/2)-1, :);
            elseif dim == 2
                out = resized(:, round(diff/2):width_d+round(diff/2)-1, :);
            end
        else
            % will horizontally pad input
            if dim == 3
                out = [max_val*ones(size(resized, 1), round(diff/2), 3), ...
                       resized, ...
                       max_val*ones(size(resized, 1), round(diff/2), 3)];
            elseif dim == 2
                out = [max_val*ones(size(resized, 1), round(diff/2)), ...
                       resized, ...
                       max_val*ones(size(resized, 1), round(diff/2))];
            end
        end
        
    end

    assert(abs(ratio_d - size(out,2)/size(out,1)) < 1e-3, 'output ratio has changed')
end
