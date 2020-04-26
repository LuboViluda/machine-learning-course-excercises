function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% compute cost
error = (X * theta) - y;
error_square = error.^2;
J = (1 / (2 * m)) * sum(error_square);

% regularization
reg_param = (lambda / (2 * m)) * sum(theta(2:end).^2);
J = J + reg_param;

grad = (1/m) * (X' * error) + ([ zeros(1, size(theta,2)); theta(2:end, :)])* (lambda / m);













% =========================================================================

grad = grad(:);

end
