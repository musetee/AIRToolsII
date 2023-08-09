%Ex6: comparison with filtered back projection (inverse Radon transform)

% Per Christian Hansen, November 19, 2012, DTU Informatics.
clear, clc

N = 32;             % Image is N-by-N.
eta = 0.03;         % Relative noise level.
theta = 5:5:180;    % Angles.
k = 100;            % Number of iterations.

% Create the phantom.
[~,~,x] = binarytomo(N); X = reshape(x,N,N);

% Compute the filtered back projection (FBP) reconstruction.
S = radon(X,theta);
E = randn(size(S));
S = S + eta*norm(S,'fro')*E/norm(E,'fro');
Xfbp = iradon(S,theta);
Xfbp = Xfbp(2:end-1,2:end-1);
Xfbp(Xfbp<0) = 0;
efbp = norm(x-Xfbp(:));  % Error in FBP reconstruction.

% Show the exact image and the FBP reconstruction.
figure(1), clf
subplot(2,2,1)
imagesc(X); colormap gray
title('Phantom'), axis image off
subplot(2,2,2)
imagesc(Xfbp), axis image off, caxis([0 1])
title('FBP reconstruction')

% Prepare for the algebraic approach: create the matrix A and the rhs b.
% Each column of A is the radon transform of a black image with a single
% white pixel; looping over all pixels in the image generates all columns
% The rhs b is just a reshaped version of the radon transform of the data.
A = zeros(numel(S),N^2);
b = S(:);
J = 0;
for i=1:N
    for j=1:N
        XX = zeros(N);
        XX(j,i) = 1;
        RR = radon(XX,theta);
        J = J+1;
        A(:,J) = RR(:);
    end
end
[A,b] = rzr(A,b);  % Remove zero rows from the system.
save Abx A b x

% Compute an ART reconstruction with nonnegativity constraints.
options.nonneg = true;
Xart = kaczmarz(A,b,1:k,[],options);
I = [1 N N^2-N+1 N^2]; Xart(I,:) = 0; % Fix errors in the corner pixels.
err = zeros(k,1);
for i=1:k
    err(i) = norm(x-Xart(:,i));  % Error history.
end
[~,kmin] = min(err);

% Plot the ART reconstruction and the error history.
subplot(2,2,3)
imagesc(reshape(Xart(:,kmin),N,N)), caxis([0 1])
axis image off, caxis([0 1])
title(['Algebraic reconst.  k = ',num2str(kmin)])
subplot(2,2,4)
plot(1:k,err,'-b',[0 k],efbp*[1 1],'--r')
title('Error history')
legend('ART','FBP')