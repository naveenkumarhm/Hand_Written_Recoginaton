function W = randInitializeWeights(L_in, L_out)
W = zeros(L_out, 1 + L_in);


e_init = 0.87;
W = rand(L_out, 1 + L_in) * 2 * e_init - e_init;
q=size(W)




% =========================================================================

end