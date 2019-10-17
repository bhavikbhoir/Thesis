function vc = blkM2vc(M, blkS) 
M = double(M)/255;
[height,width,pixel] = size(M);

r = blkS(1) ; 
c = blkS(2) ;  

if (rem(height, r) ~= 0) || (rem(width, c) ~= 0) 
    error('blocks do not fit into matrix') 
end

x = width/c; 
y = height/r; 


N   = x*y; 
rc  = r*c; 
vc  = zeros(rc, N, pixel);

for ii = 0:y - 1 
    vc(:,(1:x)+ii*x) = reshape(M((1:r)+ii*r,:),rc,x);
end
