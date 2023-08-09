%% Exercise 5: the advantage of including the non-negativity projection in the iterative algorithm
clear
%% projection
N=32;
space=5;
theta=1:space:180;
[A,bex,xex] = binarytomo(N,theta);
x_proj=reshape(xex,[N,N]);
b_proj=reshape(bex,[N*2,length(theta)]);  

%% adds a noise vector e with Gaussian white noise, scaled such that || e ||2 / || bex ||2 = 0.02
eta = 0.03;
seed = 30;
rng(seed,'twister');
s = rng;
e = randn(size(bex)); 
e = eta*norm(bex)*e/norm(e); 
b_noise = bex + e;
b_noise_proj = reshape(b_noise,[N*2,length(theta)]);

%% train lambda
options_train.nonneg=true;
kmax=100;
method=@kaczmarz;
trained_relaxpar = train_relaxpar(A,b_noise,xex,method,kmax,options_train); 