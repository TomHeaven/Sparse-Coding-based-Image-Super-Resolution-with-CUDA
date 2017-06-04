function [hIm] = ScSR_cpu(lIm, up_scale, Dh, overlap)

patch_size = sqrt(size(Dh, 1));

% bicubic interpolation of the low-resolution image
mIm = single(imresize(lIm, up_scale, 'bicubic'));
[h, w] = size(mIm);

% extract low-resolution image features
lImfea = extr_lIm_fea_V2(mIm, patch_size);

% patch indexes for sparse recovery (avoid boundary)
gridx = 3:patch_size - overlap : w-patch_size-2;
gridx = [gridx, w-patch_size-2];
gridy = 3:patch_size - overlap : h-patch_size-2;
gridy = [gridy, h-patch_size-2];

% load pre-computed base sparse vectors
load('Children_sparse_coe.mat');

% loop to recover each low-resolution patch (CPU Version)
hIm = cpuScSR(mIm, lImfea, Dh, Children_sparse_coe,gridx,gridy, patch_size);