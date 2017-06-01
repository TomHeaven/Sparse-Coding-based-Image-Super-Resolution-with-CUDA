function cropResultImages()
%% crop result images and show details
% By TomHeaven, hanlin_tan@nudt.edu.cn, 2016.10.28
close all;
%% settings

load('compareResults.mat');


% region = [175 299 233 387 ];  im_number = 2; 
region = [176 188 258 251 ];  im_number = 11; 

for i = im_number 
    h1 = figure;
    imshow(results{i}.im);
    %pause;
    
    crop_h = results{i}.im(region(1):region(3), region(2):region(4),:);
    crop_bic = results{i}.bic.im_h(region(1):region(3), region(2):region(4),:);
    crop_yang = results{i}.yang.im_h(region(1):region(3), region(2):region(4),:);
    crop_cuda = results{i}.cuda.im_h(region(1):region(3), region(2):region(4),:);
  
    imwrite(crop_h, sprintf('im%d_crop_h.bmp', i));
    imwrite(crop_bic, sprintf('im%d_crop_bic.bmp', i));
    imwrite(crop_yang, sprintf('im%d_crop_yang.bmp', i));
    imwrite(crop_cuda, sprintf('im%d_crop_cuda.bmp', i));
end


end