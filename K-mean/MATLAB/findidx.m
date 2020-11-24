function idx = findidx(X, centro)
idx = zeros(size(X,1), 1);
n = size(X,1);

for i = 1:n
    % according to k-mean algorithm's formula
    minus_term = X(i,:);
    intermediate = (centro - ones(size(centro))*diag(minus_term)).^2;
    normd = sum(intermediate,2);
    [~,idx(i)] = min(normd);
end
