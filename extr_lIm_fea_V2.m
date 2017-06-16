%% 重写特征提取函数

function [lImFea] = extr_lIm_fea_V2(lIm, patch_size)

[nrow, ncol] = size(lIm);
patch_num = (nrow-patch_size+1)*(ncol-patch_size+1);
Fea_dim = patch_size^2*4;

lImFea = zeros(Fea_dim, patch_num);
%lImFea_temp = zeros([nrow, ncol, 4]);
% first order gradient filters
hf1 = [-1,0,1];
vf1 = [-1,0,1]';
 
lImFea_1 = conv2(lIm, hf1, 'same');
lImFea_1 = im2col(lImFea_1,[patch_size,patch_size],'sliding');
lImFea_2 = conv2(lIm, vf1, 'same');
lImFea_2 = im2col(lImFea_2,[patch_size,patch_size],'sliding');

% second order gradient filters
hf2 = [1,0,-2,0,1];
vf2 = [1,0,-2,0,1]';
 
lImFea_3 = conv2(lIm,hf2,'same');
lImFea_3 = im2col(lImFea_3,[patch_size,patch_size],'sliding');
lImFea_4 = conv2(lIm,vf2,'same');
lImFea_4 = im2col(lImFea_4,[patch_size,patch_size],'sliding');

for i=1:patch_num
    lImFea(:,i) = [lImFea_1(:,i);lImFea_2(:,i);lImFea_3(:,i);lImFea_4(:,i)];
end

