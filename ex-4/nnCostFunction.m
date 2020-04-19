function [J grad] = nnCostFunction(nn_params, 
                                   input_layer_size, 
                                   hidden_layer_size, 
                                   num_labels, 
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), 
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), 
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% transfer Y to clasification form
Y = zeros(m, num_labels);
for i = 1 : m 
  Y(i, y(i)) = 1;
endfor

% forward propagation part
% add bias to the a1 (= x1)
X = [ones(m, 1) X];

% compute second/hidden layer
Z2 = X * Theta1';
A2 = sigmoid(Z2);

% add bias to the a2
A2 = [ones(m, 1) A2];

% compute output layer
Z3 = A2 * Theta2';
A3 =  sigmoid(Z3);

% cost without regularization
J = (-Y .* log(A3) - (1 - Y).* log(1 - A3));
J = 1/m * sum(sum(J));

% remove thetas related to the bias for regularization
T1 = Theta1(:, 2:end); % 25 * 400, remove thetas for bias
T2 = Theta2(:, 2:end); % 10 * 25

% add regularization for theta params
J = J + (lambda / (2 * m)) * (sum(sum(T1 .^ 2)) + sum(sum(T2 .^ 2)));

% compute back propagation
d3 = A3 - Y;

d2 = ((Theta2)' * d3')' .* A2 .* (1 - A2);

Theta2_grad = Theta2_grad + d3' * A2;  

Theta1_grad = Theta1_grad + (d2' * X)(2:end,:);

Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;

% add regularization for theta
[x1, y1] = size(Theta1_grad);
[x2, y2] = size(Theta2_grad);

Theta1_grad = Theta1_grad + (lambda/m) * [zeros(x1,1) Theta1(:,2:end)];
Theta2_grad = Theta2_grad + (lambda/m) * [zeros(x2,1) Theta2(:,2:end)];

% Unroll gradients  
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
