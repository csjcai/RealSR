clear; clc; close all; warning off;
addpath(genpath('./.'));
addpath(genpath('/home/./caffe/')) ;
caffe.set_mode_gpu();
caffe.set_device(0);

scale      = 4;
ext        =  {'*.JPG','*.png','*.bmp', '*.tif'};
CameraType = {'Canon', 'Nikon'};
folder     = './.';
filepaths  = dir(fullfile(folder, '*.caffemodel'));
weights = fullfile(folder,filepaths.name);
model = '*.prototxt';
net = caffe.Net(model, weights,'test');

count = 1;
for q = 1:2
    folder2   =  ['Test/', CameraType{q}, '/', num2str(scale)];
    filepaths2 =  [];
    for p = 1 : length(ext)
        filepaths2 = cat(1,filepaths2, dir(fullfile(folder2, ext{p})));
    end
    for i = 1 : 2: length(filepaths2)
        disp(i)
        HR = imread(fullfile(folder2,filepaths2(i).name));
        HR = modcrop(HR, 4);
        HR_Ycbcr = im2double(rgb2ycbcr(HR));
        HR_Y = im2single(HR_Ycbcr(:, :, 1));
        
        LR = imread(fullfile(folder2,filepaths2(i+1).name));
        LR = modcrop(LR, 4);
        LR_Ycbcr = im2double(rgb2ycbcr(LR));
        LR_Y = im2single(LR_Ycbcr(:, :, 1));
        
        size_img = size(LR_Y);
        if ((size_img(1) > 1200) && (size_img(2) > 1200))
            size_patch = [1200 1200];
            size_skip = [800 800];
            [patch] = im2patch(LR_Y, size_patch, size_skip);
            [patchY] = im2patch(HR_Y, size_patch, size_skip);
        else
            patch = LR_Y;
            patchY = HR_Y;
        end
        
        for k = 1:size(patch,3)
            net.blobs('data').reshape([size(patch(:,:,k),1) size(patch(:,:,k),2), 1, 1]);
            net.reshape();
            res = net.forward({patch(:,:,k)});
            result = res{1};
            
            [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(patchY(:,:,k)),im2uint8(result),0,0);
            PSNRs(count) = PSNRCur;
            SSIMs(count) = SSIMCur;
            count = count + 1;
        end
    end
end
mean(PSNRs)
mean(SSIMs)
caffe.reset_all()

