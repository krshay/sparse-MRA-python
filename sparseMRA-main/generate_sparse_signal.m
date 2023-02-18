function x = generate_sparse_signal(L, p)

x = double(rand(L,1)>(1-p));