function mssim = msssim(X, Xcomp, K, window, scale, weight, method) 

K = [0.01 0.03]; 
window = fspecial('gaussian', 11, 1.5); 
scale = 5; 
weight = [0.1448 0.2856 0.2001 0.2363 0.1333]; 
method = 'product'; 
[H1 W1] = size(win); 
min_img_width = min(H, W)/(2^(scale-1)); 
max_win_width = max(H1, W1); 
 
im1 = double(X); im2 = double(Xcomp); 

C1 = (K(1)*L)^2; 
C2 = (K(2)*L)^2; 

win = window/sum(sum(window)); 
mu1   = filter2(win, im1, 'valid'); 
mu2   = filter2(win, im2, 'valid'); 

mu1_sq = mu1.*mu1; 
mu2_sq = mu2.*mu2; 
mu1_mu2 = mu1.*mu2; 

sigma1_sq = filter2(win, im1.*im1, 'valid') - mu1_sq; 
sigma2_sq = filter2(win, im2.*im2, 'valid') - mu2_sq; 
sigma12 = filter2(win, im1.*im2, 'valid') - mu1_mu2; 

num1 = 2*mu1_mu2 + C1; 
num2 = 2*sigma12 + C2; 

den1 = mu1_sq + mu2_sq + C1; 
den2 = sigma1_sq + sigma2_sq + C2; 

ssim_map = ones(size(mu1)); 
index = (denominator1.*denominator2 > 0); 
ssim_map(index) =   (num1(index).*num2(index))./(den1(index).*den2(index)); 
index = (den1 ~= 0) & (den2 == 0); 
ssim_map(index) = num1(index)./den1(index); 
 
ssim = mean2(ssim_map); 
downsamplefilter = ones(2)./4; 

for l = 1:scale 
   [mssim_array(l) ssim_map_array{l} mcs_array(l) cs_map_array{l}] = ssim(im1, im2, K, 
   window); 
   
   filtered_im1 = imfilter(im1, downsamplefilter, 'symmetric', 'same'); 
   filtered_im2 = imfilter(im2, downsamplefilter, 'symmetric', 'same'); 
   
   im1 = filtered_im1(1:2:end, 1:2:end); 
   im2 = filtered_im2(1:2:end, 1:2:end); 
end 

if (method == 'product')    
   mssim = prod(mcs_array(1:scale-1).^weight(1:scale-1))*(mssim_array(scale).^weight(scale)); 
else 
   weight = weight./sum(weight); 
   mssim = sum(mcs_array(1:scale-1).*weight(1:scale-1)) + mssim_array(scale).*weight(scale);
end
