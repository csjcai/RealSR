Iref=(im2double(imread('test.jpg')));
tau=[0.2,0.3,0.1,0.2,200,100];
% tau=[0,0,0,0,50,100];
It=Iref;
for i=1:3
It(:,:,i)=warpImg(Iref(:,:,i),tau);
end

imshow([Iref,It])

tic
[ImTrans,tau] = align(It,Iref, zeros(6,1),2);
toc;
imshow([It,Iref,ImTrans,10*abs(ImTrans-Iref)])