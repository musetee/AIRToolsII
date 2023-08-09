%% Exercise 1-1: how to set up a test problem and solve it
clear
%% projection
N=256;
space=1;
theta=1:space:180;
[A,bex,xex] = odftomo(N,theta);
x_proj=reshape(xex,[N,N]);
b_proj=reshape(bex,[N*2,length(theta)]);  

%% adds a noise vector e with Gaussian white noise, scaled such that || e ||2 / || bex ||2 = 0.02
eta = 0.2;
e = randn(size(bex));
e = eta*norm(bex)*e/norm(e); 
b_noise = bex + e;
b_noise_proj = reshape(b_noise,[N*2,length(theta)]);  

%% ART reconstruction
% This use of the variable options enforces nonnegativity constraints on the reconstructions. 
% The output variable Xart is an array with 9 columns, and column no. k holds the k-th iteration 
% (here, one iteration is defined as one sweep through the columns of the matrix).
options.nonneg = true;
Xart = kaczmarz(A,b_noise,1:4,[],options);
Xart_last_iter=Xart(:,end);
Xart_proj = reshape(Xart_last_iter,[N,N]);  

%% cglsAIR
iteration_K=4;
X_cgls=cglsAIR(A,b_noise,iteration_K);
X_cgls_proj=reshape(X_cgls(:,end),[N,N]);  
%% plot
img_classes=3;
figure();
subplot(2,img_classes,1);
imshow(x_proj,[]);
title('origin')
subplot(2,img_classes,2);
imshow(b_proj,[]);
title('projection')
subplot(2,img_classes,3);
imshow(b_noise_proj,[]);
title('noised proj')
subplot(2,img_classes,4);
imshow(Xart_proj,[]);
title('ART recon')
subplot(2,img_classes,5);
imshow(X_cgls_proj,[]);
title('cgls recon')