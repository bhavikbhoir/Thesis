function X = vec2blkM(vec, r, h, w)  
%vec2blkM reshapes a matrix vec of rc by 1 vectors into a block-matrix X of xh by ph size  
% Each rc-element column of vec is converted into a r by c block of a matrix X and placed as a 
block-row element   
[VectorsSize ,VectorCount] = size(vec) ;  
73 
px = VectorsSize*VectorCount ;  
if ( (rem(px, h) ~= 0) || (rem(h, r) ~= 0) )  
    error('number of rows of the matrix error')  
end 
if ( (rem(px, w) ~= 0) || (rem(w, r) ~= 0) )  
    error('number of rows of the matrix erro')  
end 
ph = px/h;  
if ( (rem(VectorsSize, r) ~= 0) || (rem(VectorCount*r, h) ~= 0) )  
error('block size error')  
end  
c = VectorsSize/r ;  
xh = zeros(r, VectorCount*c);  
xh(:) = vec ;  
nrb = h/r ;  
X = zeros(h, ph);  
for ii = 0:nrb-1  
X((1:r)+ii*r, :) = xh(:, (1:ph)+ii*ph) ;  
end
