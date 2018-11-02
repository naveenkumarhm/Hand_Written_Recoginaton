clear;

inputsize  = 400;  % 20x20 Input Images of Digits
hiddensize = 25;   % 25 hidden units
outputsize = 10;          % 10 labels, from 1 to 10   
                          


% Load Training Data
fprintf('Data is loading\n')

load('data.mat');
n = size(X, 1);

% Randomly selecting  100 data points to display
slct = randperm(size(X, 1));
slct = slct(1:100);

displayData(X(slct, :));%Display the digits

fprintf('Program paused. Press enter to continue.\n');
pause;


%%  Initializing Pameters


fprintf('\nInitializing Neural Network Parameters\n')

initial_Theta1 = randInitializeWeights(inputsize, hiddensize);
initial_Theta2 = randInitializeWeights(hiddensize, outputsize);

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];



%%  Training Neural network

fprintf('\nTraining Neural Network \n')

options = optimset('MaxIter', 20);

Lambda = .2;

costFunction = @(p) CostFunct(p, inputsize,hiddensize,outputsize, X, y, Lambda);


[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);


Theta1 = reshape(nn_params(1:hiddensize * (inputsize + 1)), ...
                 hiddensize, (inputsize + 1));

Theta2 = reshape(nn_params((1 + (hiddensize * (inputsize + 1))):end), ...
                 outputsize, (hiddensize + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Predicting the digit


pred = predict(Theta1,Theta2, X);

fprintf('Precentage of predicition is correct: %f\n', mean(double(pred == y)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;




rp = randperm(n);

for i = 1:n
  
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));

    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    
   
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end

