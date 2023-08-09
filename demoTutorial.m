%demoTutorial  Demonstrates the binarytomo and odf tomo test problems

% Per Christian Hansen, May 25, 2012, DTU Informatics.
clear, clc

N = 64;                     % Problem size.
eta = 0.02;                 % Relative noise level.
whichproblem = 1;           % 1 = smooth image, 2 = binary image.

% Generate the test problem
switch whichproblem
    case 1  % Smooth image.
        k = 20;   % Number of iterations.
        [A,bex,x] = odftomo(N);
    case 2  % Binary image.
        k = 200;   % Number of iterations.
        [A,bex,x] = binarytomo(N);
    otherwise
        error('Test problem not available')
end

figure(1), clf
subplot(2,2,1)
imagesc(reshape(x,N,N)), axis image, caxis([0 1])
title('True (smooth)')

% Noisy data.
e = randn(size(bex));
e = eta*norm(bex)*e/norm(e);
b = bex + e;

% ART (Kaczmarz) with non-negativity constraints.
if k > 25
    disp(['Running ',num2str(k),' iterations, may take a while'])
end
options.nonneg = true;
Xart = kaczmarz(A,b,1:k,[],options);

% Cimmino with non-neg. constraints and Psi-2 relax. param. choice.
options.lambda = 'psi2';
Xcimmino = cimmino(A,b,1:k,[],options);

% CGLS followed by non-neg. projection.
Xcgls = cgls(A,b,1:k);
Xcgls(Xcgls<0) = 0;

% Error histories, 2-norm.
E = zeros(k,3);
for i=1:k
    E(i,1) = norm(x-Xcimmino(:,i));
    E(i,2) = norm(x-Xart(:,i));
    E(i,3) = norm(x-Xcgls(:,i));
end
figure(2), clf
plot(E,'linewidth',2)
title('Semi-convergence','fontsize',14)
legend('Cimmino','ART','CGLS')
xlabel('Iteration \itk')
ylabel('Error || x^k - x^* ||_2')

% Error histories, 1-norm.
F = zeros(k,3);
for i=1:k
    F(i,1) = norm(x-Xcimmino(:,i),1);
    F(i,2) = norm(x-Xart(:,i),1);
    F(i,3) = norm(x-Xcgls(:,i),1);
end
figure(3), clf
plot(F,'linewidth',2)
title('Semi-convergence','fontsize',14)
legend('Cimmino','ART','CGLS')
xlabel('Iteration \itk')
ylabel('Error || x^k - x^* ||_1 (sum af abs. dev.)')

figure(1)

subplot(2,2,2)
[~,kmin] = min(E(:,1));
imagesc(reshape(Xcimmino(:,kmin),N,N)), axis image, caxis([0 1])
title(['Cimmino,  k = ',num2str(kmin)])

subplot(2,2,3)
[~,kmin] = min(E(:,2));
imagesc(reshape(Xart(:,kmin),N,N)), axis image, caxis([0 1])
title(['ART,  k = ',num2str(kmin)])

subplot(2,2,4)
[~,kmin] = min(E(:,3));
imagesc(reshape(Xcgls(:,kmin),N,N)), axis image, caxis([0 1])
title(['CGLS,  k = ',num2str(kmin)])
