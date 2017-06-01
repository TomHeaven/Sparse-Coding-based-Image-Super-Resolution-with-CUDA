%% 假设低分辨率图像的大小为[h,w],则高分辨率图像的大小为：[h*up_scale,w*up_scale]=[H,W], up_scale表示放大的倍数 
% input:
%          mIm, 低分辨率图像通过三次差值后得到的与高分辨率大小一致的图像，大小为[H,W]
%          lImfea, 低分辨率特征图像, 大小为[dim_fea,(H-patch_size+1)*(W-patch_size+1)]
%          Dh, 高分辨率字典，大小为[patch_size^2, 1024]
%          Dl, 低分辨率字典，大小为[dim_fea, 1024]
%          Children_sparse_coe, 基向量的稀疏系数,大小为[1024,dim_fea]
% output:
%          hIm, 重建后高分辨率矩阵；
%

function [hIm] = cudaScSR(mIm, lImfea, Dh, Children_sparse_coe,gridx,gridy, patch_size)

[H,W] = size(mIm);

%% 转化为一维GPU数组(按行为主)
% mIm
m_width = size(mIm, 2);
gmIm = reshape(mIm', [1, size(mIm,1)*size(mIm,2)]);
gmIm = gpuArray(double(gmIm));
% imFea
glImfea = reshape(lImfea', [1, size(lImfea,1)*size(lImfea,2)]);
%size(lImfea)
glImfea = gpuArray(glImfea);
%Children_sparse_coe
gSparseCoe = reshape(Children_sparse_coe', [1, size(Children_sparse_coe, 1)*size(Children_sparse_coe, 2)]);
gSparseCoe = gpuArray(gSparseCoe);
c_width = size(Children_sparse_coe,2);
% Dh
gDh = reshape(Dh', [1, size(Dh,1) * size(Dh, 2)]);
gDh = gpuArray(gDh);
%dh_width = size(Dh, 2);

%% GPU 计算
tic;
[ghIm, gcntMat] = srCuda(gmIm, glImfea, patch_size, m_width, gSparseCoe, c_width, gDh);
fprintf('srCuda time = %f\n', toc);
%          mIm, 低分辨率图像通过三次差值后得到的与高分辨率大小一致的图像，大小为[H,W]
%          lImfea, 低分辨率特征图像, 大小为[dim_fea,(H-patch_size+1)*(W-patch_size+1)]
%          Dh, 高分辨率字典，大小为[patch_size^2, 1024]
%          Dl, 低分辨率字典，大小为[dim_fea, 1024]
%          gSparse_coe, 基向量的稀疏系数,大小为[1024,dim_fea];

%% 取回结果并处理
hIm = gather(ghIm);
cntMat = gather(gcntMat);

cntMat = reshape(cntMat, [W H])';
hIm = reshape(hIm, [W H])';

idx = (cntMat < 1);
hIm(idx) = mIm(idx);

cntMat(idx) = 1;
hIm = hIm./ double(cntMat);
hIm = uint8(hIm);

