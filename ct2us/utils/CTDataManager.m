% ========================================================================
% file name:    CTDataManager.m
% desciption:   manage CT volume dataset
% author:       Xihan Ma
% date:         2024-02-29
% ========================================================================
function [volume, volume_info, volume_name, trans_idx] = CTDataManager(dataID)

volume_data_path = '../assets/CT_raw/';
volume_file = {
                'CTLiver.nii'               % 1 3D Slicer example
                'CTLiverMsk.nii'            % 2 3D Slicer example tissue mask (from 250:end)
                's0011.nii.gz'              % 3
                's0058.nii.gz'              % 4
                's0223.nii.gz'              % 5 not usable
                's0250.nii.gz'              % 6 not usable
                's0461.nii.gz'              % 7
                's0649.nii.gz'              % 8
                's0885.nii.gz'              % 9
               };

% TODO: assign slice index range for each dataset
trans_idx_range = {
                    [250, 615]                  % 1
                    [250, 615]                  % 2
                    [40, 275]                   % 3
                    [80, 300]                   % 4
                    []                          % 5 not usable
                    [60, 285]                   % 6 not usable
                    [55, 265]                   % 7
                    [55, 230]                   % 8
                    [55, 290]                   % 9
                  };
trans_idx = trans_idx_range{dataID};

sagit_idx_range = {
                    [-inf, inf]                  % 1
                    [-inf, inf]                  % 2
                    []                  % 3
                    []                  % 4
                    []                  % 5
                    []                  % 6
                    []                  % 7
                  };

assert(dataID >= 0 && dataID <= length(volume_file))

volume = niftiread([volume_data_path, volume_file{dataID}]);
for frm = 1:size(volume, 3)
    volume(:,:,frm) = flipud(volume(:,:,frm)');
end
volume_info = niftiinfo([volume_data_path, volume_file{dataID}]);

file_name = split(volume_file{dataID}, '.');
volume_name = file_name{1};


