function analyze()
%% analyze and visualize results of compareAlgorithms
% By TomHeaven, hanlin_tan@nudt.edu.cn, 2016.07.21

close all;

% if save a previous result as compareResults_nonAtomic.mat,
% we can combine it with the current result by setting this option to true.
restoreResults = false;

%% settings
fontSize = 20; % figure font size

if restoreResults && exist('compareResults_nonAtomic.mat', 'file') == 2
    load('compareResults_nonAtomic.mat');
    results0 = results;
end

load('compareResults.mat');

len = length(results);
%len = 2; % debug

if restoreResults
    for i = 1 : len
        results{i}.bic = results0{i}.bic;
        results{i}.yang = results0{i}.yang;
        results{i}.sparse = results0{i}.sparse;
        results{i}.im = results0{i}.im;
        results{i}.im_l = results0{i}.im_l;
        results{i}.filename = results0{i}.filename;
    end
    save('compareResults.mat', 'results');
end

psnr = zeros(len, 4);
time = zeros(len, 4);

for i = 1 : len
    psnr(i, 1) = results{i}.bic.psnr;
    psnr(i, 2) = results{i}.yang.psnr;
    psnr(i, 3) = results{i}.sparse.psnr;
    psnr(i, 4) = results{i}.cuda.psnr;
    
    time(i, 1) = results{i}.bic.time;
    time(i, 2) = results{i}.yang.time;
    time(i, 3) = results{i}.sparse.time;
    time(i, 4) = results{i}.cuda.time;
end


h1 = figure;
bar(psnr);
legend('Bicubic', 'Yang', 'Sparse', 'CUDA');
title('PSNR comparison');
xlabel('Image No.');
ylabel('dB');
set(gca, 'FontSize', fontSize);
saveas(h1, 'psnr_comparison.jpg');


h2 = figure;
bar(time);
legend('Bicubic', 'Yang', 'Sparse', 'CUDA');
title('Running time comparison');
xlabel('Image No.');
ylabel('sec');
set(gca, 'FontSize', fontSize);
saveas(h2, 'time_comparison.jpg');

for i = 1 : len
    imwrite(results{i}.im_l, sprintf('im%d_im_l.bmp', i));
    imwrite(results{i}.im, sprintf('im%d_im_h.bmp', i));
    imwrite(results{i}.bic.im_h, sprintf('im%d_bic.bmp', i));
    imwrite(results{i}.yang.im_h, sprintf('im%d_yang.bmp', i));
    imwrite(results{i}.sparse.im_h, sprintf('im%d_sparse.bmp', i));
    imwrite(results{i}.cuda.im_h, sprintf('im%d_cuda.bmp', i));
end


for i = 1 : len
    res = results{i};
    sz = size(res.im);
    name = sscanf(results{i}.filename, 'dataset/%s');
    l = length(name);
    name = name(1:l - 4);
    l = length(name);
    if (name(l) == 'T' && name(l-1) == 'G')
        name = name(1:l-3);
    end
    
    fprintf('%d & %s & $%d \\times %d $ & %.2f / %.3f & %.2f / %.2f & %.2f / %.2f & %.2f / %.2f \\\\ \n', ...
        i, name, sz(1), sz(2), res.bic.psnr, res.bic.time, res.yang.psnr, res.yang.time, ...
        res.sparse.psnr, res.sparse.time, res.cuda.psnr, res.cuda.time);
end

save('analysis.mat', 'psnr', 'time');
end