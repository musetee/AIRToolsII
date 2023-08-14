%% Exercise 1-4: test different stopping rules
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
options.lbound = 0;
options.relaxpar=trained_relaxpar;
options.stoprule.type='DP';
options.stoprule.taudelta=norm(e);
[Xart_DP,info_DP]= kaczmarz(A,b_noise,1:art_k,[],options);
for k=1:size(Xart_DP,2)
    err_DP(k) = norm( xex - Xart_DP(:,k));
end

%% ART method with the NCP stopping rule
options_NCP.lbound = 0;
options_NCP.relaxpar=trained_relaxpar ;
options_NCP.stoprule.type='NCP';
options_NCP.stoprule.res_dims=[N*2,length(theta)];
[Xart_NCP,info_NCP] = kaczmarz(A,b_noise,1:art_k,[],options_NCP);
for k=1:size(Xart_NCP,2)
    err_NCP(k) = norm( xex - Xart_NCP(:,k));
end

%% plot error curves
figure(); hold on
plot(err_DP,'DisplayName','DP');
plot(err_NCP,'DisplayName','NCP');
legend()