close all; clc; clear;

addpath(genpath('./.'));
folder  = './.';  % Put the image pairs in this folder
filepaths = dir(fullfile(folder, '*.png'));

for i = 1:2:size(filepaths)
    I1  = im2double(imread(fullfile(folder,filepaths(i).name)));       % reference image
    I2  = im2double(imread(fullfile(folder,filepaths(i+1).name)));     % target image

    s = 2;                                             % s = len1/len2, len1&lend2 are the focal length of captured image (len1>len2)
    r = 1 - 1/s;                                       % Scale
    I2_zoom=warpImg(I2,[-r,0,0,0]);
    
    tau0=zeros(6,1);
    iter=3;                                            % number of iterations
    [I2_t,tau] = align_l(I2_zoom,I1,tau0,iter);
    
    [I2_t_l]=luminance_transfer(I1,I2_t);              % transfer luminance
    [I2_t_c]=color_transfer(I1,I2_t);                  % transfer color

    % where you aim to save the image
    imwrite(I1, ['./.', filepaths(i).name])            
    imwrite(I2_t_c, ['./.', filepaths(i+1).name])
end
