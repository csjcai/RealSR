function [patch] = im2patch(img, size_patch, size_skip)
% im2patch converts [Y X] size image
% first to [size_patch(1) size_patch(2) num_patch] size 3D array,
% then to [size_patch(1)*size_patch(2) num_patch] size 2D array.
patch_loc = patchLocation(size(img), size_patch, size_skip);
patch = img(patch_loc);
