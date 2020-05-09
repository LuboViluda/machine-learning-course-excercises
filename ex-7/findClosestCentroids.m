function idx = findClosestCentroids(X, centroids)

K = size(centroids, 1);
m = size(X,1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

for i = 1:m
  idx(i) = 1;
  minDistance = norm(X(i,:)-centroids(1,:))^2;
  for o = 2:K    
      distance = norm(X(i,:)-centroids(o,:))^2;
      if (distance < minDistance)
        idx(i) = o;
        minDistance = distance;
      endif
  endfor
endfor

end

