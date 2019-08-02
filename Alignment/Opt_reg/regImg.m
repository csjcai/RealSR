function [I2warp,tau,residue,OmegaOut] = regImg(I1,I2,tau,weight,maxIts)

% This function registers I2 to I1 based on different different transformation
% tau --- the transformation parameter,
% which must be initialized, since its length specifies the type of transf.
% length(tau) = 2: translation
% length(tau) = 3: rigid
% length(tau) = 4: similarity
% length(tau) = 6: affine
% length(tau) = 8: projective

% initialize variables
I1 = double(I1);
I2 = double(I2);
sizeI = size(I1);
sizeD = sizeI(1)*sizeI(2);

if nargin < 3 || isempty(tau)
    tau = zeros(6,1); % the default is affine transf.
end

if  nargin < 4 || isempty(weight)
    weight = ones(sizeD,1);
else
    weight = reshape(weight,[sizeD,1]);
end

if  nargin < 5 || isempty(maxIts)
    maxIts = 10;
end

for iter = 1:maxIts    
    % warping to calculate I2(X) = I1(f(X,tau))
    [I2warp,OmegaOut] = warpImg(I2,tau);
    % get derivatives
 
    %% compute least square solution
    y = reshape(I1-I2warp,[sizeD,1]);

    X=getJacob(I2warp,tau);
    % solve
    Weight=weight.*(OmegaOut(:)==0);
    A = X'*bsxfun(@times,X,Weight);
    dtau = (A+0.0001*diag(diag(A)))\(X'*(Weight.*y));
    tau = tau + dtau;
    % check termination condition
    if max(abs(dtau./(tau+eps))) < 0.0001;
        break
    end
end

residue = reshape( y, sizeI );
end


function X=getJacob(I,tau)

    sizeI = size(I);    
    % get derivatives
    [I2warp_x,I2warp_y] = getGradient(I); 
    % coordinates
    [yCoord,xCoord] = meshgrid(1:sizeI(2),1:sizeI(1));
    xCoord = xCoord - round(sizeI(1)/2);
    yCoord = yCoord - round(sizeI(2)/2);
    % compute X(i,:) = [I2x,I2y]*Jacob(d[x;y]/d tau)
switch length(tau)
        case 2
           X = [I2warp_x(:), I2warp_y(:)]; % 1-by-2  
        case 3
%              X = [I2warp_x(:),I2warp_y(:)]*[(-sin(tau(1))*xCoord(:)-cos(tau(1))*yCoord(:)),1,0
%                                             ( cos(tau(1))*xCoord(:)-sin(tau(1))*yCoord(:)),0,1]
            
            X = [(-sin(tau(1))*xCoord(:)-cos(tau(1))*yCoord(:)).*I2warp_x(:)+...
                ( cos(tau(1))*xCoord(:)-sin(tau(1))*yCoord(:)).*I2warp_y(:),I2warp_x(:), I2warp_y(:)]; % 1-by-3
        case 4
            X = [xCoord(:).*I2warp_x(:)+yCoord(:).*I2warp_y(:),...
                -yCoord(:).*I2warp_x(:)+xCoord(:).*I2warp_y(:),...
                I2warp_x(:), I2warp_y(:)]; % 1-by-4
        case 6
            X = [xCoord(:).*I2warp_x(:), xCoord(:).*I2warp_y(:),...
                 yCoord(:).*I2warp_x(:), yCoord(:).*I2warp_y(:),...
                 I2warp_x(:), I2warp_y(:)]; % 1-by-6
        case 8
            X = [ xCoord(:).*I2warp_x(:), xCoord(:).*I2warp_y(:), -(xCoord(:).^2).*I2warp_x(:)-(xCoord(:).*yCoord(:)).*I2warp_y(:),...
                  yCoord(:).*I2warp_x(:), yCoord(:).*I2warp_y(:), -(yCoord(:).^2).*I2warp_y(:)-(xCoord(:).*yCoord(:)).*I2warp_x(:),...
                  I2warp_x(:), I2warp_y(:)]; % 1-by-6
            D = [xCoord(:),yCoord(:)]*[tau(3);tau(6)]+1;
            X = bsxfun(@rdivide,X,D+eps);
        otherwise
end    

end

function [phi_x ,phi_y] = getGradient(phi)
%  detector = [1 -8 0 8 -1]'/12;
 detector = [-1,0 1]'/2;
phi_x = imfilter(phi,detector, 'replicate');
phi_y = imfilter(phi,detector','replicate');
end




