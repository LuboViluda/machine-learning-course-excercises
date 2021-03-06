function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% compute cost function
J = (-y' * log(sigmoid(X*theta)) - (1 - y)' * log(1 - sigmoid(X*theta)));

% cost function with regularized theta parameters
J = 1/m * sum(J) + (lambda/(2*m))* sum(theta(2:end).^2);

% prepare for compution of gradient steps with vectorication
% strip theta0 and feature x1 (by convention x1 = 1 for all i)
theta_to_reg = theta(2:end);
X_to_reg = X(:,2:end);

% compute theta0 (index 1) as no regularized
grad_theta0 = (1/m) * X(:,1)'* (sigmoid(X*theta) - y);

% note input for h(x) - sigmoid is full input (no striped theta and X (as 
% optimalization depends even on theta0 even if isn't regularized and vice versa 

% compute others theta as regularized
grad_temp = (1/m) * X_to_reg'* (sigmoid(X*theta) - y) + (lambda/(m)) * theta_to_reg;

% join results
grad = [grad_theta0; grad_temp];

% =============================================================
end
