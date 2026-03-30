function f0 = find_f0_yin(x)
% FIND_F0_YIN  Find fundamental frequency using the YIN method
%
%    F0 = FIND_F0_YIN(X)
%
%    Find the fundamental frequency of signal X using the
%    YIN method developed by de Cheveigne and Kawahara.

% $Id: find_f0_yin.m 84 2005-08-22 09:47:16Z mairas $


% range limits for f0

f0_min = 60;
f0_max = 2000;

tau_min = floor(x.fs/f0_max);
tau_max = ceil(x.fs/f0_min);

% integration window size
W = length(x.s)-tau_max;
if W < 1
    f0 = 0;
    %warning('F0-YIN: Too short window');
    return;
end

% threshold for step 4
threshold = 0.1;

% local context variables

tau_min_local = 0.8;  % search range for the local estimate
tau_max_local = 1.2;  % search range for the local estimate
r_local = 16;  % local context radius

% initialization

dn_array = zeros(r_local, tau_max);

% initial energy
E = sum(x.s.^2);

% Step 2: Difference function

% calculate the difference function
d = zeros(1, tau_max);
for tau = 1:tau_max
  d(tau) = sum((x.s(1:W) - x.s((1:W)+tau)).^2);
end

% Step 3: Cumulative mean normalized difference function

dn = d./(cumsum(d)./(1:tau_max));

% Step 4: Absolute threshold

% find the minima below the threshold
diff_dn = diff(dn);
diff1 = diff_dn(tau_min-1:end-1);
diff2 = diff_dn(tau_min:end);
dn_range = dn(tau_min:end-1);
minima = (diff1 <= 0) & (diff2 > 0) & (dn_range <= threshold);

% get the first such minimum
[val,idx] = max(minima);
% if not found, use the global minimum instead
if isempty(val) || (val==0)
  [min_val, min_idx] = min(dn_range);
  min_idx = min_idx + tau_min - 1;
else
  min_idx = idx + tau_min - 1;
  min_val = dn(min_idx);
end


% Step 5: Parabolic interpolation
if ~isempty(min_idx) && isscalar(min_idx) && min_idx > 1 && min_idx < length(dn)
    [min_idx_interp,min_val_interp] = interp_min(dn((min_idx-1):(min_idx+1)),min_idx);
else
    % Use direct value without interpolation
    if isempty(min_idx) || ~isscalar(min_idx)
        min_idx = tau_min;  % fallback to minimum tau
    end
    min_idx_interp = min_idx;
    min_val_interp = dn(min_idx);
end

% Step 6: Best local estimate not practical for short frames, so omitted


% final result:

f0 = x.fs/min_idx_interp;




function [x0,y0] = interp_min(y,min_idx)
% get a minimum value by parabolically interpolating three points


% Ensure y is a column vector
y = y(:);

%A = [1 -1 1; 0 0 1; 1 1 1];
%Ai = A^-1;
Ai = [0.5 -1 0.5; -0.5 0 0.5; 0 1 0];

a_ = [0.5 -1 0.5];
b_ = [-0.5 0 0.5];

a = Ai(1,:)*y;
b = Ai(2,:)*y;
c = Ai(3,:)*y;

x0 = -b/(2*a);
y0 = a*x0^2+b*x0+c;
x0 = min_idx + x0;
