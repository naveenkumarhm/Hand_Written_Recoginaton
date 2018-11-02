function [J grad] = CostFunc(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


m = size(X, 1);
      
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%% Part 1 implementation

A1 = [ones(m, 1) X];

z2 = A1 * Theta1';
A2 = sigmoid(z2);
A2 = [ones(size(A2,1), 1) A2];

A3 = A2 * Theta2';
H = sigmoid(A3);
Hsum = H;

yVec = zeros(m,num_labels);

for i = 1:m
    yVec(i,y(i)) = 1;
end


J = 1/m * sum(sum(-1 * yVec .* log(Hsum)-(1-yVec) .* log(1-Hsum)));


%% Part 2 implementation



for t = 1:m

	% For the input layer, where l=1:
	A1 = [1; X(t,:)'];

	% For the hidden layers, where l=2:
	z2 = Theta1 * A1;
	A2 = [1; sigmoid(z2)];

	A3 = Theta2 * A2;
	a3 = sigmoid(A3);

	yy = ([1:num_labels]==y(t))';
    
	delta_3 = a3 - yy;

	delta_2 = (Theta2' * delta_3) .* [1; sigmoidGradient(z2)];
	delta_2 = delta_2(2:end); 
   

	Theta1_grad = Theta1_grad + delta_2 * A1';
	Theta2_grad = Theta2_grad + delta_3 * A2';
end

Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];



grad = [Theta1_grad(:) ; Theta2_grad(:)];


end