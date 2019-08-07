% function [img] = patch2im(patch, size_img, size_patch, size_skip)
% 
% img = zeros(size_img);
% w = zeros(size_img);
% patch_loc = patchLocation(size_img, size_patch, size_skip);
% 
% for n=1:size(patch_loc,3)
%     img(patch_loc(:,:,n)) = img(patch_loc(:,:,n)) + patch(:,:,n);
%     w(patch_loc(:,:,n)) = w(patch_loc(:,:,n)) + 1;
% end
% img = img ./ w;

function [img] = patch2im(patch, size_img, size_patch, size_skip)

img = zeros(size_img);
w = zeros(size_img);
patch_loc = patchLocation(size_img, size_patch, size_skip);

for n=1:size(patch_loc,3)
    img(patch_loc(1:end-1,1:end-1,n)) = img(patch_loc(1:end-1,1:end-1,n)) + patch(1:end-1,1:end-1,n);
    w(patch_loc(1:end-1,1:end-1,n)) = w(patch_loc(1:end-1,1:end-1,n)) + 1;
end
img = img ./ w;