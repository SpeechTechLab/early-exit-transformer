function a = mytrlp(frame, p, a_prev, lambda_factor) 
    if nargin < 4
        lambda_factor = 0.1; % Default regularization factor relative to energy
    end
    
    lambda2 = 0.9; % Forgetting factor for previous coefficients
    
    % Autocorrelation method
    R_full = xcorr(frame, p);
    R = R_full(p+1:end);
    
    % Energy of the frame (R(0))
    energy = R(1);
    
    % Normalize regularization by energy to make it scale-invariant
    lambda1 = lambda_factor * energy;
    
    R_matrix = toeplitz(R(1:p));
    r_vector = R(2:p+1);
    
    % Regularization term
    % (R + lambda1 * I) * a = r + lambda1 * lambda2 * a_prev
    a_reg = lambda2 * a_prev;
    R_reg = R_matrix + lambda1 * eye(p);
    r_reg = r_vector + lambda1 * a_reg;
    
    % Solve for new LPC coefficients
    a = R_reg \ r_reg;
end