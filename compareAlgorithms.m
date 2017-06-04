function compareAlgorithms()
%% entrance function
% By TomHeaven, hanlin_tan@nudt.edu.cn, 2016.07.21

%% settings
restoreResults = false;
dataDir  = 'dataset/';
start = 1;

% Enable cpu version algorithms. However, this process will take a few hours.
% You can set this option to false to evaluate the CUDA version algorithm
% only.
enableSP = false;

% init GPU
gpuDevice(1);

% load dictionary
load('Dictionary/D_1024_0.15_5.mat');

if restoreResults && exist('compareResults.mat', 'file') == 2
    load('compareResults.mat');
end

fileExt = '*.bmp';
files = dir(fullfile(dataDir, fileExt));
len = size(files,1);
results = cell(len, 1);
%len = 2; % debug

for i=start:len
    fileName = strcat(dataDir, files(i,1).name);
    fprintf('\nProcessing image %d, path = %s\n', i, fileName);
    im = imread(fileName);
    im_l = imresize(im, 0.5);
    res = compare(im_l, im, Dl, Dh, enableSP);
    res.filename = fileName;
    
    if enableSP
        results{i} = res;
    else
        results{i}.cuda = res.cuda;
    end
    save('compareResults.mat', 'results');
end

analyze;

end



function res = compare(im_l, im, Dl, Dh, enableSP)

%% result structure
res.im_l = im_l;
res.im =im;

%% set parameters
lambda = 0.8;                   % sparsity regularization
overlap = 4;                    % the more overlap the better (patch size 5x5)
up_scale = 2;                   % scaling factor, depending on the trained dictionary
maxIter = 20;                   % if 0, do not use backprojection

a = find(im_l > 255);
im_l(a) = 255;
% load dictionary
load('Dictionary/D_1024_0.15_5.mat');

if size(im_l, 3) == 3
    % change color space, work on illuminance only
    im_l_ycbcr = rgb2ycbcr(uint8(im_l));
    im_l_y = im_l_ycbcr(:, :, 1);
    im_l_cb = im_l_ycbcr(:, :, 2);
    im_l_cr = im_l_ycbcr(:, :, 3);
else
    im_l_y = im_l;
end


%% 1. start timing for sparse cuda super resolution
enableCUDA = true;
if enableCUDA
    tic;
    % image super-resolution based on sparse representation
    [im_h_y] = ScSR_gpu(im_l_y, up_scale, Dh, overlap);
    [im_h_y] = backprojection(im_h_y, im_l_y, maxIter);
    res.cuda.time = toc;
    
    % upscale the chrominance simply by "bicubic"
    [nrow, ncol] = size(im_h_y);
    if size(im_l, 3) == 3
        im_h_cb = imresize(im_l_cb, [nrow, ncol], 'bicubic');
        im_h_cr = imresize(im_l_cr, [nrow, ncol], 'bicubic');
        
        im_h_ycbcr = zeros([nrow, ncol, 3]);
        im_h_ycbcr(:, :, 1) = im_h_y;
        im_h_ycbcr(:, :, 2) = im_h_cb;
        im_h_ycbcr(:, :, 3) = im_h_cr;
        im_h = ycbcr2rgb(uint8(im_h_ycbcr));
    else
        im_h = im_h_y;
    end
    
    if size(im,1) ~= size(im_h,1) || size(im,2) ~= size(im_h,2)
        im_h = imresize(im_h, [size(im, 1) size(im, 2)]);
    end
    % compute PSNR for the illuminance channel
    sp_rmse = compute_rmse(im, im_h);
    sp_psnr = 20*log10(255/sp_rmse);
    res.cuda.im_h = im_h;
    res.cuda.psnr = sp_psnr;
end

%% 2. start timing for sparse cpu super resolution
if  enableSP
    tic;
    % image super-resolution based on sparse representation
    [im_h_y] = ScSR_cpu(im_l_y, up_scale, Dh, overlap);
    [im_h_y] = backprojection(im_h_y, im_l_y, maxIter);
    res.sparse.time = toc;
    
    % upscale the chrominance simply by "bicubic"
    if size(im_l, 3) == 3
        [nrow, ncol] = size(im_h_y);
        im_h_cb = imresize(im_l_cb, [nrow, ncol], 'bicubic');
        im_h_cr = imresize(im_l_cr, [nrow, ncol], 'bicubic');
        
        im_h_ycbcr = zeros([nrow, ncol, 3]);
        im_h_ycbcr(:, :, 1) = im_h_y;
        im_h_ycbcr(:, :, 2) = im_h_cb;
        im_h_ycbcr(:, :, 3) = im_h_cr;
        im_h = ycbcr2rgb(uint8(im_h_ycbcr));
    else
        im_h = im_h_y;
    end
    
    
    if size(im,1) ~= size(im_h,1) || size(im,2) ~= size(im_h,2)
        im_h = imresize(im_h, [size(im, 1) size(im, 2)]);
    end
    % compute psnr
    sp_rmse = compute_rmse(im, im_h);
    
    
    
    sp_psnr = 20*log10(255/sp_rmse);
    res.sparse.im_h = im_h;
    res.sparse.psnr = sp_psnr;
end

%% 3. start timing for Yang's super resolution
if enableSP
    tic;
    % image super-resolution based on sparse representation
    [im_h_y] = ScSR_yang(im_l_y, up_scale, Dh, Dl, lambda, overlap);
    [im_h_y] = backprojection(im_h_y, im_l_y, maxIter);
    res.yang.time = toc;
    
    % upscale the chrominance simply by "bicubic"
    if size(im_l, 3) == 3
        [nrow, ncol] = size(im_h_y);
        im_h_cb = imresize(im_l_cb, [nrow, ncol], 'bicubic');
        im_h_cr = imresize(im_l_cr, [nrow, ncol], 'bicubic');
        
        im_h_ycbcr = zeros([nrow, ncol, 3]);
        im_h_ycbcr(:, :, 1) = im_h_y;
        im_h_ycbcr(:, :, 2) = im_h_cb;
        im_h_ycbcr(:, :, 3) = im_h_cr;
        im_h = ycbcr2rgb(uint8(im_h_ycbcr));
    else
        im_h = im_h_y;
    end
    
    if size(im,1) ~= size(im_h,1) || size(im,2) ~= size(im_h,2)
        im_h = imresize(im_h, [size(im, 1) size(im, 2)]);
    end
    % compute psnr
    sp_rmse = compute_rmse(im, im_h);
    sp_psnr = 20*log10(255/sp_rmse);
    res.yang.im_h = im_h;
    res.yang.psnr = sp_psnr;
end


%% 4. bicubic interpolation for reference
tic;
[nrow, ncol] = size(im_h_y);
im_b = imresize(im_l, [nrow, ncol], 'bicubic');
res.bic.time = toc;
res.bic.im_h = im_b;

if size(im,1) ~= size(im_b,1) || size(im,2) ~= size(im_b,2)
    im_b = imresize(im_b, [size(im, 1) size(im, 2)]);
end
% compute PSNR for the illuminance channel
bb_rmse = compute_rmse(im, im_b);
bb_psnr = 20*log10(255/bb_rmse);
res.bic.psnr = bb_psnr;

end