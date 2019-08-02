function  [Itt,para]=color_transfer(Ic,It,ind)


[m1,n1,c]=size(Ic);
X=reshape(Ic,[m1*n1,c]);

[m2,n2,c]=size(It);
Y=reshape(It,[m2*n2,c]);

if nargin <3
   ind=[1:(m2*n2)]';  
end


mu_x=mean(X);

S_x=(X-repmat(mu_x,[m1*n1,1]))'*(X-repmat(mu_x,[m1*n1,1]))/(m1*n1);
[Ux,Dx,~]=svd(S_x);
mu_y=mean(Y(ind,:));
S_y=(Y(ind,:)-repmat(mu_y,[size(ind,1),1]))'*(Y(ind,:)-repmat(mu_y,[size(ind,1),1]))/(size(ind,1));
[Uy,Dy,~]=svd(S_y);

A=Ux*diag(diag(Dx).^(0.5))*Ux'*Uy*diag(diag(Dy).^(-0.5))*Uy';
b=mu_x'-A*mu_y';


Z=A*Y'+repmat(b,[1,m2*n2]);
Z=Z';
para.A=A;
para.b=b;
%mu_z=mean(Z);
%S_z=(Z-repmat(mu_z,[m2*n2,1]))'*(Z-repmat(mu_z,[m2*n2,1]))/(m2*n2);
Itt=reshape(Z,[m2,n2,c]);
end

