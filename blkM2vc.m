function vec = blkM2vec(X, blkS)  
[h,w,p] = size(X); 

r = blkS(1) ; 
c = blkS(2) ;   

if (rem(h, r) ~= 0) || (rem(w, c) ~= 0)  
    error('blocks do not fit into matrix')  
end 

x = w/c; 
y = h/r; 
N = x*y; 
rc = r*c;  

vc  = zeros(rc, N, p); 

for ii = 0:y - 1  
    vc(:,(1:x)+ii*x) = reshape(X((1:r)+ii*r,:),rc,x); 
end 
 
