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

%% ART method with the Discrepancy Principle stopping criterion
art_k=50;
options.relaxpar=trained_relaxpar;
[Xart_1,info_1]= kaczmarz(A,b_noise,1:art_k,[],options);
Xart_1(Xart_1 < 0) = 0;
for k=1:size(Xart_1,2)
    err_1(k) = norm( xex - Xart_1(:,k));
end

%% ART method with the Discrepancy Principle stopping criterion
options_2.relaxpar=trained_relaxpar;
options_2.lbound = 0;
[Xart_2,info_2]= kaczmarz(A,b_noise,1:art_k,[],options_2);
for k=1:size(Xart_2,2)
    err_2(k) = norm( xex - Xart_2(:,k));
end

%% plot error curves
figure(); hold on
plot(err_1,'DisplayName','no nonneg');
plot(err_2,'DisplayName','nonneg');
legend()