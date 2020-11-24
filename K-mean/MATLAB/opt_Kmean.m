function centr = opt_Kmean(X, ini_centr, max_iter)

[m n] = size(X); 
K = size(ini_centr, 1);
centr = ini_centr;
% Iteratively compute Centroids.
for i=1:max_iter
    % Assign index for each point.
    idx = zeros(size(X,1), 1);
    n = size(X,1);
    for i = 1:n 
        minus_term = X(i,:);
        intermediate = (centr - ones(size(centr))*diag(minus_term)).^2;
        normd = sum(intermediate,2);
        [~,idx(i)] = min(normd);
    end
    % Compute new centroids given indices
    [m n] = size(X);
    centroids = zeros(K, n);
    for i = 1:K
        test = X;
        indices = find(idx ~= i);
        test(indices,:) = 0;
        centroids(i,:) = 1/(m-length(indices))*sum(test);
    end
end
