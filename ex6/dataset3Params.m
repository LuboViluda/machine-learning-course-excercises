function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

%all_C = [ 0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30 ];
%all_sigma = [ 0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30 ];

%smallest_error = 1;

%for i = 1:8
%  temp_sigma = all_sigma(i,1);
%  for o = 1:8
%    fprintf('looping for:');
%    i 
%    o 
    
%    temp_C = all_C(o,1);
%    model= svmTrain(X, y, temp_C, @(x1, x2) gaussianKernel(x1, x2, temp_sigma));
%    predictions = svmPredict(model, Xval);
%    current_error = mean(double(predictions ~= yval))
%    smallest_error
%    if (smallest_error > current_error)
%      fprintf('found new better params, prediction: \n');
%      smallest_error = current_error;
%      C = temp_C;
%      sigma = temp_sigma;
%   endif
%  endfor
%endfor

%C
%sigma

end
