function s_v = valid(s)
% VALID  Return the valid part of the signal.

% Check if valid is a simple flag or a region definition
if isscalar(s.valid) && (s.valid == 0 || s.valid == 1)
    % Simple flag - just return the signal as-is
    s_v = s;
    return;
end

% Original code for region-based validity
s_ = s.s;
t_ = s.time;
if s.valid(1)==Inf
  warning('Region of validity not defined.');
  s_v = s;
else
  t1 = s.time(s.valid(1));
  if length(s.valid)>1
    t2 = s.time(s.valid(2));
  else
    t2 = s.time(end);
  end
  s_v = trim(s,t1,t2);
end