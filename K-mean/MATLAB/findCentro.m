function centroids = findCentro(X, idx, K)

[m n] = size(X);
centroids = zeros(K, n);

for i = 1:K
    % setup variables
    test = X;
    indices = find(idx ~= i);
    test(indices,:) = 0;
    % Assign new centroid
    centroids(i,:) = 1/(m-length(indices))*sum(test);
end



