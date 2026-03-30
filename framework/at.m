function I = at(x,t)
% AT  Return element index corresponding to a time value.
%
% I = AT(X,T)  Return element index I at time location t in signal
% object X.

% $Id: at.m 119 2006-09-26 12:28:25Z mairas $

%I = at(x.time.t,t);
I = round(x.time.fs*(t-x.time.begin))+1;
