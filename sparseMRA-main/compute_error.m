function err = compute_error(x_est, x_true)

% this function computes the relative error with respect to dihedral group
% (shifts + reflection) times Z_2 (sign change).

X_true = fft(x_true);
X_est = fft(x_est);
a1 = abs(ifft(X_true.*X_est)); % the abs values takes also the sign change into account
a2 = abs(ifft(X_true.*conj(X_est))); % the reflected signal
[max_correlation, ~] = max([a1; a2]);
err = norm(x_est).^2 + norm(x_true).^2 - 2*max_correlation;
err = err/norm(x_true).^2; % relative error

end
