Iref=(im2double(imread('test.jpg')));
tau=[0.2,0.3,0.1,0.2,200,100];
% tau=[0,0,0,0,50,100];
It=Iref;
for i=1:3
It(:,:,i)=warpImg(Iref(:,:,i),tau);
end
It=1.4*abs(It+0.0001)+0.1;
imshow([Iref,It])

tic
[ImTrans,tau,ImTrans_c] = align_c(It,Iref, zeros(6,1),2);
toc;
imshow([It,Iref,ImTrans,ImTrans_c,10*abs(ImTrans_c-Iref)])