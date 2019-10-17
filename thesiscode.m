%Here is a case where we analyze an input image at level 3 wavelet decomposition

tic;  %Get the run-time of the code

clc;
close all;
clear all;

%Input the image
O=imread('Veggies.jpg');                                               %Read Input image 
X1 = O(:,:,1); X2 = O(:,:,2); X3 = O(:,:,3);                              %Extract the RGB Channels 
[rw cl p] = size(O);                              
wavelet=input('Select the wavelet ','s');                                 %Select wavelet for analysis 
 
%Level 3 decomposition of the image 
[c1,l1] = wavedec2(X1,3,wavelet); 
[c2,l2] = wavedec2(X2,3,wavelet);  
[c3,l3] = wavedec2(X3,3,wavelet); 
 
%To reconstruct the level 1,2,3 approximation from C and S 
A31 = wrcoef2('a',c1,l1,wavelet,3); 
A21 = wrcoef2('a',c1,l1,wavelet,2); 
A11 = wrcoef2('a',c1,l1,wavelet,1);
H11 = wrcoef2('h',c1,l1,wavelet,1);  
V11 = wrcoef2('v',c1,l1,wavelet,1); 
D11 = wrcoef2('d',c1,l1,wavelet,1);  
H21 = wrcoef2('h',c1,l1,wavelet,2);
V21 = wrcoef2('v',c1,l1,wavelet,2);  
D21 = wrcoef2('d',c1,l1,wavelet,3); 
H31 = wrcoef2('h',c1,l1,wavelet,3); 
V31 = wrcoef2('v',c1,l1,wavelet,3); 
D31 = wrcoef2('d',c1,l1,wavelet,3);           
A32 = wrcoef2('a',c2,l2,wavelet,3); 
A22 = wrcoef2('a',c2,l2,wavelet,2); 
A12 = wrcoef2('a',c2,l2,wavelet,1); 
H12 = wrcoef2('h',c2,l2,wavelet,1);  
V12 = wrcoef2('v',c2,l2,wavelet,1); 
D12 = wrcoef2('d',c2,l2,wavelet,1);  
H22 = wrcoef2('h',c2,l2,wavelet,2); 
V22 = wrcoef2('v',c2,l2,wavelet,2);  
 
D22 = wrcoef2('d',c2,l2,wavelet,3); 
H32 = wrcoef2('h',c2,l2,wavelet,3); 
V32 = wrcoef2('v',c2,l2,wavelet,3); 
D32 = wrcoef2('d',c2,l2,wavelet,3);        
A33 = wrcoef2('a',c3,l3,wavelet,3); 
A23 = wrcoef2('a',c3,l3,wavelet,2); 
A13 = wrcoef2('a',c3,l3,wavelet,1); 
H13 = wrcoef2('h',c3,l3,wavelet,1);  
V13 = wrcoef2('v',c3,l3,wavelet,1);
D13 = wrcoef2('d',c3,l3,wavelet,1);  
H23 = wrcoef2('h',c3,l3,wavelet,2); 
V23 = wrcoef2('v',c3,l3,wavelet,2);  
D23 = wrcoef2('d',c3,l3,wavelet,3); 
H33 = wrcoef2('h',c3,l3,wavelet,3); 
V33 = wrcoef2('v',c3,l3,wavelet,3); 
D33 = wrcoef2('d',c3,l3,wavelet,3); 
             
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
              
L31 = imresize([imresize([a3_cod1,d3_hcod1;d3_vcod1,d3_dcod1],size(d2_hcod1),'bilinear'),d2_hcod1;d2_vcod1,d2_dcod1],size(d1_hcod1),'bilinear'),d1_hcod1;d1_vcod1,d1_dcod1]; 
L32 = [imresize([imresize([a3_cod2,d3_hcod2;d3_vcod2,d3_dcod2],size(d2_hcod2),'bilinear'),d2_hcod2;d2_vcod2,d2_dcod2],size(d1_hcod2),'bilinear'),d1_hcod2;d1_vcod2,d1_dcod2]; 
L33 = [imresize([imresize([a3_cod3,d3_hcod3;d3_vcod3,d3_dcod3],size(d2_hcod3),'bilinear'),d2_hcod3;d2_vcod3,d2_dcod3],size(d1_hcod3),'bilinear'),d1_hcod3;d1_vcod3,d1_dcod3]; 
             
L3(:,:,1) = L31;  
L3(:,:,2) = L32;  
L3(:,:,3) = L33;   
image(uint8(L3)); 
axis image; 
title('Level 3 decomposition'); 
 
% Calculate the default parameters and perform the actual compression. 
[thres,sorh,nkeep] = ddencmp('cmp','wv',X1); 
[Xcomp1,CXC1,LXC1,Perf01,PerfL21]=wdencmp('gbl',c1,l1,wavelet,3,thres,sorh,nkeep); 
[thres,sorh,nkeep] = ddencmp('cmp','wv',X2); 
[Xcomp2,CXC2,LXC2,Perf02,PerfL22]=wdencmp('gbl',c2,l2,wavelet,3,thres,sorh,nkeep); 
[thres,sorh,nkeep] = ddencmp('cmp','wv',X3); 
[Xcomp3,CXC3,LXC3,Perf03,PerfL23]= wdencmp('gbl',c3,l3,wavelet,3,thres,sorh,nkeep);          

Xcomp(:,:,1) = Xcomp1; 
Xcomp(:,:,2) = Xcomp2; 
Xcomp(:,:,3) = Xcomp3;         

%Image Compression using Neural Networks 
r=input('enter input neurons ');                      %Set input neurons  
h=input('enter hidden layer neurons ');               %Set hidden layer neurons 
X1=blkM2vec(X1,[r r]);                                %Block-matrix M to vector count 
X2=blkM2vec(X2,[r r]);  
X3=blkM2vec(X3,[r r]);  
net_c = feedforwardnet(1,’trainlm’);                  %Initialize the Feed Forward Neural network 
net_c.layers{1}.size = h; 
net_c.trainparam.epochs=input('enter epochs ');       %Set the training parameters 
net_c.trainparam.goal=0;  
[net_s,tr]=train(net_c,X1,X1);  
s=sim(net_s,X1);                                      %Simulate the neural networks 
C1=vec2blkM(a,r,rw,cl); 
s=sim(net_s,X2);  
C2=vec2blkM(a,r,rw,cl); 
s=sim(net_s,X3);  
C3 =vec2blkM(a,r,rw,cl); 
Ncomp = cat(3,C1,C2,C3);                              %compute the compressed image  

subplot(1,3,1); image(uint8(O)); title('Original Image');
axis square
subplot(1,3,2); 
image(uint8(Xcomp)); title('Wavelet Compressed Image');
axis square
subplot(1,3,3); 
image(uint8(NComp)); title({'Hybrid (Wavelet Transform - Neural Network)';' Compressed Image'});
axis square
            
%Calculating Performance Parameters 
%To find the compression ratio 
in = imfinfo('original.jpg'); 
disp(in); 
out = imfinfo('compressed.jpg'); 
disp(out); 
ib=in.FileSize; 
disp('The file size of the original image is'); 
disp(ib);  
cb=out.FileSize; 
disp('The file size of the Compressed image is'); 
disp(cb);  
cr=ib/cb; 
disp('The compression ratio is'); 
disp(cr); 
  
% Calculate mean square error and power signal to noise ratio             
mse=(sum(sum(sum((X-Xcomp).*(X-Xcomp)))))/(rw*cl*p); 
PSNR=10*log10((255^2)/mse); 
disp('Mean square error = '); 
disp(mse); 
disp('PSNR ='); 
disp(PSNR); 

%Calculate Structural Similarity 
%Single Scale 
ssim_value = ssim(X, RGBImage); 
disp('SSIM value is:'); 
disp(ssim_value(:)); 

%Multi Scale 
ms = msssim(X,Xcomp); 
msssim = mean2(ms);             
disp('MS-SSIM value is:'); 
disp(msssim); 

%Measurement of Error Sensitivity 
I1 = X(:,:,1);                                    %Extract the RGB channels
I2 = X(:,:,2); 
I3 = X(:,:,3);                                
R1 = Xcomp(:,:,1); 
R2 = Xcomp(:,:,2); 
R3 = Xcomp(:,:,3); 
 
D1 = abs(double(I1)-double(R1)).^2; 
Er = (sum(sum(sum(D1(:)))))/(RS1.*RS2.*3); 
D2 = abs(double(I2)-double(R2)).^2; 
Eg = (sum(sum(sum(D2(:)))))/(RS1.*RS2.*3); 
D3 = abs(double(I3)-double(R3)).^2; 
Eb = (sum(sum(sum(D3(:)))))/(RS1.*RS2.*3); 
 
PSNRes=10*log10(3/(Er+Eg+Eb)); 
disp('PSNR error sensitivity is'); 
disp(PSNRes); 
 
%Measurement of Structural Distortion 
%Red Component 
Xai = mean(I1(:)); 
[maxvalXpi,idx]=max(I1, [], 1); 
[row,col,pix]=ind2sub(size(I1), idx); 
[minvalXbi,idx]=min(I1, [], 1); 
[row,col,pix]=ind2sub(size(I1), idx); 
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
 
PSNRsd=10*log10(3/(Sr+Sg+Sb)); 
disp('PSNR structural dist is'); 
disp(PSNRsd); 
  
% Measurement of Edge distortion 
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

%Measurement of Quality PSNR 
PSnRq = (0.32*PSNRes) + (0.38*PSNRed) + (0.3*PSNRsd); 
disp('Quality PSNR is'); 
disp(PSnRq); 

toc;                          %Calculate Run-time 

