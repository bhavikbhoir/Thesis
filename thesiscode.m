tic;
clc;
close all;
clear all;
%Input the image
X=imread('D:\Downloads\Photos\Veggies.jpg');
RS1 =512; RS2 = 512;
X = imresize (X, [RS1 RS2]);
[ipH ipW ipp] = size(X);
disp(size(X));
X1 = X(:,:,1);                      %Extract the RGB channels
X2 = X(:,:,2);
X3 = X(:,:,3);
imwrite(X,'wavein.jpg');
string = 'wavein.jpg';
[rw cl p] = size(X);


%Displaying the original image
figure; 
NbColors = 255;
map = gray(NbColors);
image(X);colormap(map); title('Original image');colorbar;

wavelet=input('Select the wavelet ','s'); 
level=input('Select the level of decomposition ');

           %To perform a level 3 decomposition of the image
            [C1,S1] = wavedec2(X1,3,wavelet); 
            [C2,S2] = wavedec2(X2,3,wavelet); 
            [C3,S3] = wavedec2(X3,3,wavelet); 
            
            %To reconstruct the level 1,2,3 approximation from C and S
            A31 = wrcoef2('a',C1,S1,wavelet,3);
            A21 = wrcoef2('a',C1,S1,wavelet,2);
            A11 = wrcoef2('a',C1,S1,wavelet,1);
            H11 = wrcoef2('h',C1,S1,wavelet,1); 
            V11 = wrcoef2('v',C1,S1,wavelet,1); 
            D11 = wrcoef2('d',C1,S1,wavelet,1); 
            H21 = wrcoef2('h',C1,S1,wavelet,2);
            V21 = wrcoef2('v',C1,S1,wavelet,2); 
            D21 = wrcoef2('d',C1,S1,wavelet,3);
            H31 = wrcoef2('h',C1,S1,wavelet,3);
            V31 = wrcoef2('v',C1,S1,wavelet,3); 
            D31 = wrcoef2('d',C1,S1,wavelet,3);
            
            A32 = wrcoef2('a',C2,S2,wavelet,3);
            A22 = wrcoef2('a',C2,S2,wavelet,2);
            A12 = wrcoef2('a',C2,S2,wavelet,1);
            H12 = wrcoef2('h',C2,S2,wavelet,1); 
            V12 = wrcoef2('v',C2,S2,wavelet,1); 
            D12 = wrcoef2('d',C2,S2,wavelet,1); 
            H22 = wrcoef2('h',C2,S2,wavelet,2);
            V22 = wrcoef2('v',C2,S2,wavelet,2); 
            D22 = wrcoef2('d',C2,S2,wavelet,3);
            H32 = wrcoef2('h',C2,S2,wavelet,3);
            V32 = wrcoef2('v',C2,S2,wavelet,3); 
            D32 = wrcoef2('d',C2,S2,wavelet,3);
            
            A33 = wrcoef2('a',C3,S3,wavelet,3);
            A23 = wrcoef2('a',C3,S3,wavelet,2);
            A13 = wrcoef2('a',C3,S3,wavelet,1);
            H13 = wrcoef2('h',C3,S3,wavelet,1); 
            V13 = wrcoef2('v',C3,S3,wavelet,1); 
            D13 = wrcoef2('d',C3,S3,wavelet,1); 
            H23 = wrcoef2('h',C3,S3,wavelet,2);
            V23 = wrcoef2('v',C3,S3,wavelet,2); 
            D23 = wrcoef2('d',C3,S3,wavelet,3);
            H33 = wrcoef2('h',C3,S3,wavelet,3);
            V33 = wrcoef2('v',C3,S3,wavelet,3); 
            D33 = wrcoef2('d',C3,S3,wavelet,3);
            
            A1(:,:,1) = A11;
            A1(:,:,2) = A12;
            A1(:,:,3) = A13;
            H1(:,:,1) = H11;
            H1(:,:,2) = H12;
            H1(:,:,3) = H13;
            V1(:,:,1) = V11;
            V1(:,:,2) = V12;
            V1(:,:,3) = V13;
            D1(:,:,1) = D11;
            D1(:,:,2) = D12;
            D1(:,:,3) = D13;
            A2(:,:,1) = A21;
            A2(:,:,2) = A22;
            A2(:,:,3) = A23;
            H2(:,:,1) = H21;
            H2(:,:,2) = H22;
            H2(:,:,3) = H23;
            V2(:,:,1) = V21;
            V2(:,:,2) = V22;
            V2(:,:,3) = V23;
            D2(:,:,1) = D21;
            D2(:,:,2) = D22;
            D2(:,:,3) = D23;
            A3(:,:,1) = A31;
            A3(:,:,2) = A32;
            A3(:,:,3) = A33;
            H3(:,:,1) = H31;
            H3(:,:,2) = H32;
            H3(:,:,3) = H33;
            V3(:,:,1) = V31;
            V3(:,:,2) = V32;
            V3(:,:,3) = V33;
            D3(:,:,1) = D31;
            D3(:,:,2) = D32;
            D3(:,:,3) = D33;
            
            colormap(map);
            subplot(3,4,1);image(uint8(wcodemat(A1,192)));
            title('Approximation A1')
            subplot(3,4,2);image(uint8(wcodemat(H1,192)));
            title('Horizontal Detail H1')
            subplot(3,4,3);image(uint8(wcodemat(V1,192)));
            title('Vertical Detail V1')
            subplot(3,4,4);image(uint8(wcodemat(D1,192)));
            title('Diagonal Detail D1')
            subplot(3,4,5);image(uint8(wcodemat(A2,192)));
            title('Approximation A2')
            subplot(3,4,6);image(uint8(wcodemat(H2,192)));
            title('Horizontal Detail H2')
            subplot(3,4,7);image(uint8(wcodemat(V2,192)));
            title('Vertical Detail V2')
            subplot(3,4,8);image(uint8(wcodemat(D2,192)));
            title('Diagonal Detail D2')
            subplot(3,4,9);image(uint8(wcodemat(A3,192)));
            title('Approximation A3')
            subplot(3,4,10);image(uint8(wcodemat(H3,192)));
            title('Horizontal Detail H3')
            subplot(3,4,11);image(uint8(wcodemat(V3,192)));
            title('Vertical Detail V3')
            subplot(3,4,12);image(uint8(wcodemat(D3,192)));
            title('Diagonal Detail D3')
         
            
            subplot(1,1,1,'replace')
            colors = size(unique(X));
            grayLevels = colors(1);
            
            %1st level coefficient coding
            a1_cod1 = wcodemat(A11,grayLevels);
            d1_hcod1 = wcodemat(H11,grayLevels);
            d1_vcod1 = wcodemat(V11,grayLevels);
            d1_dcod1 = wcodemat(D11,grayLevels);
            
            a1_cod2 = wcodemat(A12,grayLevels);
            d1_hcod2 = wcodemat(H12,grayLevels);
            d1_vcod2 = wcodemat(V12,grayLevels);
            d1_dcod2 = wcodemat(D12,grayLevels);
           
            a1_cod3 = wcodemat(A13,grayLevels);
            d1_hcod3 = wcodemat(H13,grayLevels);
            d1_vcod3 = wcodemat(V13,grayLevels);
            d1_dcod3 = wcodemat(D13,grayLevels);
                       
            %2nd level coefficient coding
            a2_cod1 = wcodemat(A21,grayLevels);
            d2_hcod1 = wcodemat(H21,grayLevels);
            d2_vcod1 = wcodemat(V21,grayLevels);
            d2_dcod1 = wcodemat(D21,grayLevels);

            a2_cod2 = wcodemat(A22,grayLevels);
            d2_hcod2 = wcodemat(H22,grayLevels);
            d2_vcod2 = wcodemat(V22,grayLevels);
            d2_dcod2 = wcodemat(D22,grayLevels);
            
            a2_cod3 = wcodemat(A23,grayLevels);
            d2_hcod3 = wcodemat(H23,grayLevels);
            d2_vcod3 = wcodemat(V23,grayLevels);
            d2_dcod3 = wcodemat(D23,grayLevels);
            
            % 3rd level coefficients coding
            a3_cod1 = wcodemat(A31,grayLevels);
            d3_hcod1 = wcodemat(H31,grayLevels);
            d3_vcod1 = wcodemat(V31,grayLevels);
            d3_dcod1 = wcodemat(D31,grayLevels);
            
            a3_cod2 = wcodemat(A32,grayLevels);
            d3_hcod2 = wcodemat(H32,grayLevels);
            d3_vcod2 = wcodemat(V32,grayLevels);
            d3_dcod2 = wcodemat(D32,grayLevels);
            
            a3_cod3 = wcodemat(A33,grayLevels);
            d3_hcod3 = wcodemat(H33,grayLevels);
            d3_vcod3 = wcodemat(V33,grayLevels);
            d3_dcod3 = wcodemat(D33,grayLevels);
            
            L31 = [imresize([imresize([a3_cod1,d3_hcod1;d3_vcod1,d3_dcod1],size(d2_hcod1),'bilinear'),d2_hcod1;d2_vcod1,d2_dcod1],size(d1_hcod1),'bilinear'),d1_hcod1;d1_vcod1,d1_dcod1];
            L32 = [imresize([imresize([a3_cod2,d3_hcod2;d3_vcod2,d3_dcod2],size(d2_hcod2),'bilinear'),d2_hcod2;d2_vcod2,d2_dcod2],size(d1_hcod2),'bilinear'),d1_hcod2;d1_vcod2,d1_dcod2];
            L33 = [imresize([imresize([a3_cod3,d3_hcod3;d3_vcod3,d3_dcod3],size(d2_hcod3),'bilinear'),d2_hcod3;d2_vcod3,d2_dcod3],size(d1_hcod3),'bilinear'),d1_hcod3;d1_vcod3,d1_dcod3];
            
            L3(:,:,1) = L31;
            L3(:,:,2) = L32;
            L3(:,:,3) = L33;
            
            image(uint8(L3));
            axis image;
            title('Level 3 decomposition');
            
            % Compress the image and display it. 
            % To compress the original image X, use the ddencmp command to calculate the default parameters
            % and the wdencmp command to perform the actual compression.
            [thr,sorh,keepapp] = ddencmp('cmp','wv',X1);
            [Xcomp1,CXC1,LXC1,PERF01,PERFL21] = wdencmp('gbl',C1,S1,wavelet,3,thr,sorh,keepapp);
            [thr,sorh,keepapp] = ddencmp('cmp','wv',X2);
            [Xcomp2,CXC2,LXC2,PERF02,PERFL22] = wdencmp('gbl',C2,S2,wavelet,3,thr,sorh,keepapp);
            [thr,sorh,keepapp] = ddencmp('cmp','wv',X3);
            [Xcomp3,CXC3,LXC3,PERF03,PERFL23] = wdencmp('gbl',C3,S3,wavelet,3,thr,sorh,keepapp);     
            Xcomp(:,:,1) = Xcomp1;
            Xcomp(:,:,2) = Xcomp2;
            Xcomp(:,:,3) = Xcomp3;
            NC1 = wthcoef2('t',C1,S1,3,thr,'s');
            NC2 = wthcoef2('t',C2,S2,3,thr,'s');
            NC3 = wthcoef2('t',C3,S3,3,thr,'s');
            A(:,:,1) = a3_cod1;
            A(:,:,2) = a3_cod2;
            A(:,:,3) = a3_cod3;
            
            
            X01 = waverec2(C1,S1,wavelet);
            X02 = waverec2(C2,S2,wavelet);
            X03 = waverec2(C3,S3,wavelet);
            
            X0(:,:,1) = X01;
            X0(:,:,2) = X02;
            X0(:,:,3) = X03;   

[height,width,pixel] = size(Xcomp);    
X1 = Xcomp(:,:,1);
X2 = Xcomp(:,:,2);
X3 = Xcomp(:,:,3);

r=input('enter input neurons ');                                %Set input neurons = r*r
h=input('enter hidden layer neurons ');                                %Set hidden layer neurons

blks = [r r];

I1=blkM2vcnn(X1,blks);               %Block-matrix M to vector count
I2=blkM2vcnn(X2,blks); 
I3=blkM2vcnn(X3,blks); 

net_c = feedforwardnet(1,'trainlm');  %FeedForward Neural network
net_c.layers{1}.size = h;
epochs = input('Enter epochs');
net_c.trainparam.epochs=epochs;        %Set the training parameterss
net_c.trainparam.goal=0;    
net_c.trainparam.min_grad=1e-10;
net_c.trainParam.max_fail=6;
net_c.trainParam.mu=0.01;
net_c.trainParam.mu_dec=0.1;
net_c.trainParam.mu_inc=10;
net_c.trainParam,mu_max=1e10;
net_c.trainParam.show=25;
net_c.trainParam.showWindow=true;
[net_s,tr]=train(net_c,I1,I1); 
a=sim(net_s,I1);                    %simulate the neural networks
I1_compressed=vc2blkM(a,r,height,width);
a=sim(net_s,I2); 
I2_compressed=vc2blkM(a,r,height,width);
a=sim(net_s,I3); 
I3_compressed=vc2blkM(a,r,height,width);
RGBImage = cat(3,I1_compressed,I2_compressed,I3_compressed);    %get the compressed image 
RGBImagee = uint8(RGBImage);
blks = [r r];
I1=blkM2vc(X1,blks);               %Block-matrix M to vector count
I2=blkM2vc(X2,blks); 
I3=blkM2vc(X3,blks); 
net_c = feedforwardnet(1,'trainlm');  %FeedForward Neural network
net_c.layers{1}.size = h;
net_c.trainparam.epochs=epochs;        %Set the training parameterss
net_c.trainparam.goal=0;    
net_c.trainparam.min_grad=1e-10;
% net_c.trainParam.max_fail=6;
% net_c.trainParam.mu=0.01;
% net_c.trainParam.mu_dec=0.1;
% net_c.trainParam.mu_inc=10;
% net_c.trainParam,mu_max=1e10;
% net_c.trainParam.show=25;
net_c.trainParam.showWindow=false;
[net_s,tr]=train(net_c,I1,I1); 
a=sim(net_s,I1);I1_compressed=vc2blkM(a,r,height,width);
a=sim(net_s,I2);I2_compressed=vc2blkM(a,r,height,width);
a=sim(net_s,I3); I3_compressed=vc2blkM(a,r,height,width);
RGBImage = cat(3,I1_compressed,I2_compressed,I3_compressed);    %get the compressed image 
RGBImage = uint8(RGBImage);
NbColors = 64;
map = gray(NbColors);
colormap(map);
            


subplot(1,3,1); image(uint8(X)); title('Original Image');
            axis square
            subplot(1,3,2); 
            image(uint8(Xcomp)); title('Wavelet Compressed Image');
            axis square
            subplot(1,3,3); 
            image(uint8(RGBImage)); title({'Hybrid 1 (Wavelet Transform - Neural Network)';' Compressed Image'});
            axis square
            
imwrite(RGBImagee,'nnout.jpg');
infoin = imfinfo(string);
         
            bin = infoin.FileSize;
            infoout = imfinfo('nnout.jpg');
            bout = infoout.FileSize;
            cr= bin/bout;
            disp('Compression Ratio =');
            disp(cr);

%             X = double(X);
%             RGBImage = double(RGBImage);
            mse=(sum(sum(sum((X-RGBImage).*(X-RGBImage)))))/(height*width*pixel);            
PSNR=20*log10(255/sqrt(mse));
            disp('Mean square error = ');
            disp(mse);
            disp('PSNR =');
            disp(PSNR);
            
       
            %Calculate Structural Similarity
ref = double(X);
dist = double(RGBImage);
    
sd = 1.5;
C1 = 0.01^2;
C2 = 0.03^2;
C3 = C2/2;
hRef = fspecial('gaussian',[11 11],sd);
hDis = fspecial('gaussian',[11 11],sd);
   
muRef = imfilter(ref,hRef,'conv','replicate');
muDis = imfilter(dist,hDis,'conv','replicate');
    
mux2 = muRef.^2;
muy2 = muDis.^2;
muxy = muRef.*muDis;
          
sigma01 = imfilter((ref.^2), hRef,'conv','replicate') - (mux2);
sigma02 = imfilter((dist.^2), hDis,'conv','replicate') - (muy2);    
sigma12 = imfilter((ref.*dist), hRef,'conv','replicate') - (muxy);
    
ssimNum = ((2*muxy) + C1).*((2*sigma12) + C2);
ssimDen = ((mux2) + (muy2) + C1).*(sigma01 + sigma02 + C2);
    
ssim_value = mean2(mean(ssimNum./ssimDen));
disp('SSIM value is:');
disp(ssim_value(:));
     
%MS-SSIM level =3
 X = double(X);
RGBImage = double(RGBImage);
im1 = X(:,:,1);             
im2 = RGBImage(:,:,1);             
msr = msssim(im1,im2);
msr = real(msr);

im1 = X(:,:,2);
im2 = RGBImage(:,:,2);
msg = msssim(im1,im2);
msg = real(msg);

im1 = X(:,:,3);
im2 = RGBImage(:,:,3);
msb = msssim(im1,im2);
msb = real(msb);

ms = (msr + msg + msb)/3;
msssim = mean2(ms + mean(ssimNum./ssimDen)); 
% msssim = mean2(ms);            
fprintf('The level 3 MSSSIM value is %0.4f.\n',msssim);

%MSSSIM level = 5 

im1 = X(:,:,1);             
im2 = RGBImage(:,:,1);             
msr = msssim1(im1,im2);
msr = real(msr);

im1 = X(:,:,2);
im2 = RGBImage(:,:,2);
msg = msssim1(im1,im2);
msg = real(msg);

im1 = X(:,:,3);
im2 = RGBImage(:,:,3);
msb = msssim1(im1,im2);
msb = real(msb);

ms = (msr + msg + msb)/3;
% msssim1 = mean2(ms);   
         msssim1 = mean2(ms + mean(ssimNum./ssimDen)); 
fprintf('The level 5 MSSSIM value is %0.4f.\n',msssim1);

%Error Sensitivity
r= input('Select block size');     

I1 = X(:,:,1);                      %Extract the RGB channels
I2 = X(:,:,2);
I3 = X(:,:,3);

I1=blkM2vc(I1,[r r]);               %Block-matrix M to vector count
I2=blkM2vc(I2,[r r]); 
I3=blkM2vc(I3,[r r]); 

R1 = RGBImage(:,:,1);                      %Extract the RGB channels
R2 = RGBImage(:,:,2);
R3 = RGBImage(:,:,3);
            
R1=blkM2vc(R1,[r r]);               %Block-matrix M to vector count
R2=blkM2vc(R2,[r r]); 
R3=blkM2vc(R3,[r r]);

D1 = abs(double(I1)-double(R1)).^2;
Er  = (sum(sum(sum(D1(:)))))/(RS1.*RS2.*3);
disp('Er is');
disp(Er);

D2 = abs(double(I2)-double(R2)).^2;
Eg  = (sum(sum(sum(D2(:)))))/(RS1.*RS2.*3);
disp('Eg is');
disp(Eg);

D3 = abs(double(I3)-double(R3)).^2;
Eb  = (sum(sum(sum(D3(:)))))/(RS1.*RS2.*3);
disp('Eb is');
disp(Eb);

PSNRes=10*log10(3/(Er+Eg+Eb));
disp('PSNR error sensitivity is');
disp(PSNRes);

%Structural Distortion
%Red Component
Xai = mean(I1(:));
[maxvalXpi,idx]=max(I1, [], 1);
[row,col]=ind2sub(size(I1), idx);
[minvalXbi,idx]=min(I1, [], 1);
[row,col]=ind2sub(size(I1), idx);

Yai = mean(R1(:));
[maxvalYpi,idx]=max(R1, [], 1);
[row,col]=ind2sub(size(R1), idx);
[minvalYbi,idx]=min(R1, [], 1);
[row,col]=ind2sub(size(R1), idx);

i1=0.5*((Xai - Yai)^2) 
i2=0.25*((maxvalXpi - maxvalYpi).^2) 
i3=0.25*((minvalXbi - minvalYbi).^2) 
i = i1 + i2 + i3;

Sr = sum(i(:))/RS1;
disp('Sr is');
disp(Sr);

%Green component
GXai = mean(I2(:));
[maxvalGXpi,idx]=max(I2);
[row,col,pix]=ind2sub(size(I2), idx);
[minvalGXbi,idx]=min(R2);
[row,col,pix]=ind2sub(size(R2), idx);

GYai = mean(R2(:));
[maxvalGYpi,idx]=max(R2);
[row,col,pix]=ind2sub(size(R2), idx);
[minvalGYbi,idx]=min(R3);
[row,col,pix]=ind2sub(size(R3), idx);

Gi1=0.5*((GXai - GYai)^2) 
Gi2=0.25*((maxvalGXpi - maxvalGYpi).^2) 
Gi3=0.25*((minvalGXbi - minvalGYbi).^2) 
Gi = Gi1 + Gi2 + Gi3;

Sg = sum(Gi(:))/RS1;
disp('Sg is');
disp(Sg);

%Blue component
BXai = mean(I3(:));
[maxvalBXpi,idx]=max(I3);
[row,col,pix]=ind2sub(size(I3), idx);
[minvalBXbi,idx]=min(I3);
[row,col,pix]=ind2sub(size(I3), idx);

BYai = mean(R3(:));
[maxvalBYpi,idx]=max(R3);
[row,col,pix]=ind2sub(size(R3), idx);
[minvalBYbi,idx]=min(R3);
[row,col,pix]=ind2sub(size(R3), idx);

Bi1=0.5*((BXai - BYai).^2) 
Bi2=0.25*((maxvalBXpi - maxvalBYpi).^2) 
Bi3=0.25*((minvalBXbi - minvalBYbi).^2) 
Bi = Bi1 + Bi2 + Bi3;

Sb = sum(Bi(:))/RS1;
disp('Sb is');
disp(Sb);

PSNRsd=10*log10(3/(Sr+Sg+Sb));
disp('PSNR structural dist is');
disp(PSNRsd);

% Edge distortion
[x y z]=size(X);
if z==1
    rslt=edge(X,'canny');
elseif z==3
    img1=rgb2ycbcr(X);
    dx1=edge(img1(:,:,1),'canny');
    dx1=(dx1*255);
    img2(:,:,1)=dx1;
    img2(:,:,2)=img1(:,:,2);
    img2(:,:,3)=img1(:,:,3);
    rslt=ycbcr2rgb(uint8(img2));
end
C=rslt;

C1 = C(:,:,1);                      %Extract the RGB channels
C2 = C(:,:,2);
C3 = C(:,:,3);

C1=blkM2vc(C1,[r r]);               %Block-matrix M to vector count
C2=blkM2vc(C2,[r r]); 
C3=blkM2vc(C3,[r r]); 

[a b s]=size(RGBImage);
if s==1
    rslt1=edge(RGBImage,'canny');
elseif s==3
    img11=rgb2ycbcr(RGBImage);
    dx11=edge(img11(:,:,1),'canny');
    dx11=(dx11*255);
    img22(:,:,1)=dx11;
    img22(:,:,2)=img11(:,:,2);
    img22(:,:,3)=img11(:,:,3);
    rslt1=ycbcr2rgb(uint8(img22));
end
R=rslt1;


C11 = R(:,:,1);                      %Extract the RGB channels
C22 = R(:,:,2);
C33 = R(:,:,3);

C11=blkM2vc(C11,[r r]);               %Block-matrix M to vector count
C22=blkM2vc(C22,[r r]); 
C33=blkM2vc(C33,[r r]); 

Dr = abs(double(C1)-double(C11)).^2;
EDr  = (sum(sum(sum(Dr(:)))))/(RS1.*RS2.*3);
disp('EDr is');
disp(EDr);

Dg = abs(double(C2)-double(C22)).^2;
EDg  = (sum(sum(sum(Dg(:)))))/(RS1.*RS2.*3);
disp('EDg is');
disp(EDg);

Db = abs(double(C3)-double(C33)).^2;
EDb  = (sum(sum(sum(Db(:)))))/(RS1.*RS2.*3);
disp('EDb is');
disp(EDb);

PSNRed=10*log10(3/(EDr+EDg+EDb));
disp('PSNR edge distortion is');
disp(PSNRed);

%Modified PSNR
PSNRq = (0.32*PSNRes) + (0.38*PSNRed) + (0.3*PSNRsd);
disp('modified PSNR is');
disp(PSNRq);
            toc;

