function [all_theta] = oneVsAll(X, y, num_labels, lambda)

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

options = optimset('GradObj', 'on', 'MaxIter', 50);

% for each class compute as one vs all
for i = 1:num_labels
    initial_theta = zeros(n + 1, 1);
    c = i;
    
    % 0 mapped as 10
    % compute as oneVsAll problem
    [theta] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options);  
    
    % copy results to the corresponing row
    all_theta(i,:) = theta';
endfor

end
