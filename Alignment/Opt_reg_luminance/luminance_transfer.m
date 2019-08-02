function [L_s_new,lu]=luminance_transfer(Ic,Is,ind)
if nargin<4
    model =1;
end

if nargin <3
     ind=1:size(Is(:),1);   
end

   L_c=Ic(:); L_s=Is(:);
   
   lu(1)=std(L_c)/std(L_s(ind));
   lu(2)=mean(L_c)-lu(1)*mean(L_s(ind));
   %L_s_new=(L_s-mean(L_s(ind)))*std(L_c)/std(L_s(ind))+mean(L_c);
   L_s_new=lu(1)*L_s+lu(2);
   L_s_new=reshape(L_s_new,size(Is));
    
    




end









