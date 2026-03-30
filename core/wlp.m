function [ a, e_ar ] = wlp( x, w, p )
% Compute the p-order all-pole filter coefficients of the spectral envelope of
% frame x with Weighted Linear Prediction, with weight function w.

N = length(x);

Sn = zeros(p,N+p);

for n = 1:N+p
    for i = 1:p
        if n-i > 0 && n-i <= N
            Sn(i,n) = x(n-i);
        end
    end
end

% Calculate R matrix
R = zeros(p,p);
r = zeros(p,1);
x2 = [x(:); zeros(p,1)];
for i = 1:N+p
    R = R + w(i).*Sn(:,i)*Sn(:,i)';
    r = r + w(i).*Sn(:,i).*x2(i);
end

a = R\r;
a = [1; -a];

e_ar = sum(filter(a,1,x).^2);

end

