%% 假设低分辨率图像的大小为[h,w],则高分辨率图像的大小为：[h*up_scale,w*up_scale]=[H,W], up_scale表示放大的倍数 
% input:
%          mIm, 低分辨率图像通过三次差值后得到的与高分辨率大小一致的图像，大小为[H,W]
%          lImfea, 低分辨率特征图像, 大小为[dim_fea,(H-patch_size+1)*(W-patch_size+1)]
%          Dh, 高分辨率字典，大小为[patch_size^2, 1024]
%          Dl, 低分辨率字典，大小为[dim_fea, 1024]
%          Children_sparse_coe, 基向量的稀疏系数,大小为[1024,dim_fea]
% output:
%          hIm, 重建后高分辨率矩阵；

function [hIm] = cpuScSR(mIm, lImfea, Dh, Children_sparse_coe,gridx,gridy, patch_size)
cnt = 0;
[H,W] = size(mIm);
hIm = zeros(H,W);
cntMat = zeros(H,W);


 for jj = 1:length(gridy),
     for ii = 1:length(gridx),
        
        cnt = cnt+1;
        
        xx = gridx(ii);
        yy = gridy(jj);
        
        mPatch = mIm(yy:yy+patch_size-1, xx:xx+patch_size-1);
        mMean = mean(mPatch(:));
        mPatch = mPatch(:) - mMean;
        mNorm = sqrt(sum(mPatch.^2));
        
        index_PatchFea = (xx-1)*(H-patch_size+1) + yy;
        mPatchFea = lImfea(:,index_PatchFea);   
        mfNorm = sqrt(sum(mPatchFea.^2));
        
        if mfNorm > 1,
            y = mPatchFea./mfNorm;
        else
            y = mPatchFea;
        end
        
        %Data Separation---Matrix version
         y = repmat(y',size(Children_sparse_coe,1),1);
         w = sum(y.* Children_sparse_coe,2);
        
        % generate the high resolution patch and scale the contrast
        hPatch = Dh*w;
        hPatch = lin_scale(hPatch, mNorm);
        
        hPatch = reshape(hPatch, [patch_size, patch_size]);
        hPatch = hPatch + mMean;
        
        hIm(yy:yy+patch_size-1, xx:xx+patch_size-1) = hIm(yy:yy+patch_size-1, xx:xx+patch_size-1) + hPatch;
        cntMat(yy:yy+patch_size-1, xx:xx+patch_size-1) = cntMat(yy:yy+patch_size-1, xx:xx+patch_size-1) + 1;
    end
end

idx = (cntMat < 1);
hIm(idx) = mIm(idx);

cntMat(idx) = 1;
hIm = hIm./cntMat;
hIm = uint8(hIm);

