function [patch_loc] = patchLocation(imagesize, size_patch, size_skip)

y = 1:size_skip(1):imagesize(1)-size_patch(1)+1;
x = 1:size_skip(2):imagesize(2)-size_patch(2)+1;

y = [y imagesize(1)-size_patch(1)+1];
x = [x imagesize(2)-size_patch(2)+1];

[Y,X] = meshgrid(y,x);
[dY,dX] = meshgrid(0:size_patch(1)-1,0:size_patch(2)-1);
dY = repmat(reshape(Y,[1 1 size(Y(:),1)]), [size_patch(1) size_patch(2) 1]) + repmat(dY, [1 1 size(Y(:),1)]);
dX = repmat(reshape(X,[1 1 size(X(:),1)]), [size_patch(1) size_patch(2) 1]) + repmat(dX, [1 1 size(X(:),1)]);
patch_loc = dY+(dX-1)*imagesize(1);