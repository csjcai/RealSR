function [img2warp,img2warp_color,tau] = regMGNC_c(img1,img2,tau,numLevel,maxiter)

img1 = double(img1);
img2 = double(img2);

if nargin < 3 || isempty(tau)
    tau = zeros(6,1);
end
if nargin < 4 || isempty(numLevel)
    numLevel = fix(log(size(Im,1)*size(Im,2))/log(2)/2-3);
end
if nargin < 5 || isempty(maxiter)
    maxiter = 2;
end

for l = (numLevel):-1:0
    %% construct pyramid
    I1 = imresize(img1,0.5^l,'Antialiasing',true);
    I2 = imresize(img2,0.5^l,'Antialiasing',true);   
    % initialize
    if l == numLevel
        weight = ones(size(I1));
        tau(end-1:end) = 0.5^(l-1)*tau(end-1:end); % pass from the initial value
    else
         weight = imresize(weight,[size(I1,1),size(I1,2)]);
        tau(end-1:end) = 2*tau(end-1:end); % pass from previous level
    end    
       %% gradually nonconvexity       
        % reweighted least square
    for iter = 1:(maxiter+fix(numLevel/2))
            tau_old = tau;
            [img2warp,img2warp_color,tau,residue] = regImg_c(I1,I2,tau_old,weight,1);
            weight = 1./(abs(residue)+0.1); 
%             weight = weight/(max(weight(:))+0.01);     
            if max(abs((tau_old-tau)./(tau+eps))) < 0.0001
                break
            end
    end                
end

end

