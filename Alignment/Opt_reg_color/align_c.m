function [ImTrans,tau,ImTrans_color] = align_c(I2,I1,tau,iter)

c=size(I2,3);
ImTrans = I2;
if nargin<3
  tau = zeros(6,1);
end
if nargin<4
  iter = 5;
end

numLevel = fix(log(size(I2,1)*size(I2,2))/log(2)/2-3)-1;

% I1=mean(Imref,3);
% I2=mean(Im,3);
% I2=luminance_transfer(I1,I2);
[ImTrans,ImTrans_color,tau] = regMGNC_c(I1,I2,tau,numLevel,iter);

end
