function [psnr, ssim]  = compute_diff(imGT, imSR, SF)
imGT = modcrop(imGT, SF);
% =========================================================================
% Retrieve only luminance channel
% =========================================================================
if size(imSR,3)>1
    imSR = rgb2ycbcr(imSR);
    imSR = imSR(:,:,1);
end

if size(imGT,3)>1
    imGT = rgb2ycbcr(imGT);
    imGT = imGT(:,:,1);
end
% =========================================================================
% Remove border pixels as some methods (e.g., A+) do not predict border pixels
% =========================================================================
imSR        = shave(imSR, [SF, SF]);
imGT        = shave(imGT, [SF, SF]); 

% Convert to double (with dynamic range 255)
imSR        = double(imSR); 
imGT        = double(imGT); 

% =========================================================================
% Compute Peak signal-to-noise ratio (PSNR)
% =========================================================================
% mse = mean(mean((imSR - imGT).^2, 1), 2);
imdff = imSR - imGT;
imdff = imdff(:);
rmse = sqrt(mean(imdff.^2));
psnr = 20*log10(255/rmse);
% =========================================================================
% Compute Structural similarity index (SSIM index)
% =========================================================================
[ssim, ~] = ssim_index(imSR, imGT);

% =========================================================================
% Compute information fidelity criterion (IFC)
% =========================================================================
% ifc = ifcvec(imgt, im);

end