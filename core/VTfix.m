function A_new = VTfix(A)
% A_NEW = VTFIX(A)
%
%

A = A(:).';
epsilon = 10^(-15);
% Checking for minimum phase and roots on the real axis
% Find roots
[Z,P,K] = tf2zp(1,A);

% Mirror any poles outside of the unit circle
P(abs(P) > 1) = 1./P(abs(P) > 1);

% Remove positive real poles
P = P(P < 0 | (abs(imag(P)) > epsilon));

% Reconstruct filter
[Num, Den] = zp2tf(Z,P,K);

A_new = Den;
