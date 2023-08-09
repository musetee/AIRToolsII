%% Exercise 1-3: the influence of the relaxation parameter
clear
%% projection
N=32;
space=1;
theta=1:space:180;
[A,bex,xex] = binarytomo(N,theta);
x_proj=reshape(xex,[N,N]);
b_proj=reshape(bex,[N*2,length(theta)]);  

%% adds a noise vector e with Gaussian white noise, scaled such that || e ||2 / || bex ||2 = 0.02
eta = 0.3;
e = randn(size(bex)); 
e = eta*norm(bex)*e/norm(e); 
b_noise = bex + e;
b_noise_proj = reshape(b_noise,[N*2,length(theta)]);

%% ART reconstruction
lambda=[0.05,0.1,0.15,0.2,0.3,0.5,0.7];
len_labmda=length(lambda);
art_k=50;
for j=1:len_labmda
    options.nonneg = true;
    options.relaxpar=lambda(j);
    Xart_current_lambda = kaczmarz(A,b_noise,1:art_k,[],options);
    Xart{j}=Xart_current_lambda;
    Xart_proj{j} = reshape(Xart_current_lambda(:,end),[N,N]); 

    % error-history
    for k=1:art_k
        err(k) = norm( xex - Xart_current_lambda(:,k) );
    end
    err_lambda{j}=err;
end

%% plot all errors for all lambda
figure(); hold on
for m=1:len_labmda 
    plot(err_lambda{m},'DisplayName',num2str(lambda(m))); %,
end
hold off
legend

%% train lambda
options_train.nonneg=true;
kmax=100;
method=@kaczmarz;
trained_relaxpar = train_relaxpar(A,b_noise,xex,method,kmax,options_train); 
%'landweber','cimmino','cav','drop','sart'
%'columnaction','kaczmarz','randkaczmarz','symkaczmarz'