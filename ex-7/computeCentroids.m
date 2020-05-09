function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);

% not so effective...
for i = 1:K
  sum = zeros(1,n);
  count = 0;
  for o = 1:m
    if (idx(o) == i) 
      sum = sum + X(o,:);
      count = count + 1;
    endif
  endfor

  if count != 0
    centroids(i,:) = sum / count;
  else 
    centroids(i,:) = sum;
  endif  
endfor

end

