function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values

% You need to return the following variables correctly 
J = 0;

m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% for loop version
%for i = 1:m
%    T = X (i, :);
 %   R = y (i, 1);
 %   diff = (T * theta) - R;
%    J = J + diff.^2;
%endfor
%J = (1/(2*m)) * J ; 

% vectorication
error = (X * theta) - y;
error_square = error.^2;
J = (1 / (2 * m)) * sum(error_square);

% =========================================================================

end
