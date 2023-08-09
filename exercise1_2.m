%% Exercise 1-2: semi-convergence of ART and CGLS
clear
%% projection
N=256;
space=1;
theta=1:space:180;
[A,bex,xex] = binarytomo(N,theta);
x_proj=reshape(xex,[N,N]);
b_proj=reshape(bex,[N*2,length(theta)]);  

%% adds a noise vector e with Gaussian white noise, scaled such that || e ||2 / || bex ||2 = 0.02
eta = 0.03;
e = randn(size(bex)); 
e = eta*norm(bex)*e/norm(e); 
b_noise = bex + e;
b_noise_proj = reshape(b_noise,[N*2,length(theta)]);

%% ART reconstruction
art_k=100;
options.nonneg = true;
Xart = kaczmarz(A,b_noise,1:art_k,[],options);
Xart_last_iter=Xart(:,end);
Xart_proj = reshape(Xart_last_iter,[N,N]); 

%% error-history
for k=1:art_k
    err(k) = norm( xex - Xart(:,k) );
end

%% cglsAIR
error_cgls=zeros(art_k);
X_cgls=cglsAIR(A,b_noise,cgls_k);
X_cgls(X_cgls<0)=0;
X_cgls_proj=reshape(X_cgls(:,end),[N,N]); 
for k=1:art_k 
    err_cgls(k) = norm( xex - X_cgls(:,k));
end

%% plot error history
figure()
plot(err)
hold on
plot(err_cgls)