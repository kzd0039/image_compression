clear ; close all; clc

A = double(imread('sample.jpg'));
A = A / 255; % normalize each element
img_size = size(A);
X = reshape(A, img_size(1) * img_size(2), 3);
n = 0;
kinds = [8,16,32];
savesize = [img_size,length(kinds)];
com = zeros(savesize);

for i = kinds
K = i;
n = n +1;
max_iter = 20;

% Randomly reorder the indices of examples
randidx = randperm(size(X, 1));
% Take the first K examples as centroids
ini_centr = X(randidx(1:K), :);

% Run K-Means
centroids = opt_Kmean(X, ini_centr, max_iter);

%% Compress the Image 
fprintf('Applying K-Means to compress the image...\n');

idx = findidx(X, centroids);


X_compress = centroids(idx,:);
X_compress = reshape(X_compress, img_size(1), img_size(2), 3);

com(:,:,:,n) = X_compress;


end

%% wew
% Display the original image 
subplot(2, 2, 1);
imagesc(A); 
title('Original Picture');

% Display compressed image side by side
subplot(2, 2, 2);
imagesc(com(:,:,:,1))
title(sprintf('Compressed Picture with %d Colors', 8));

% Display the original image 
subplot(2, 2, 3);
imagesc(com(:,:,:,2)); 
title(sprintf('Compressed Picture with %d Colors', 16));

% Display compressed image side by side
subplot(2, 2, 4);
imagesc(com(:,:,:,3))
title(sprintf('Compressed Picture with %d Colors', 32));
