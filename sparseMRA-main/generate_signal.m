function x = generate_signal(L, K, gauss_width)

x = zeros(L, 1);
for i = 1:K
x = x + (gaussmf(1:L,[gauss_width, randi(L)]))';
end