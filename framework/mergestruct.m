function c = mergestruct(first,varargin)
% MERGESTRUCT Merge two or more structures.
%
% C = MERGESTRUCT(A,B,...)
% Merge structures so that C contains all fields of A and B.
% If A and B have common fields, the values of latter are used.

% $Id: mergestruct.m 3 2004-02-04 12:57:04Z mairas $


c = first;
for i=1:length(varargin)
  c = mergetwo(c,varargin{i});
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Merge two structures

function c = mergetwo(a,b)

c = a;
bf=fieldnames(b);
for i=1:length(bf)
  c.(bf{i}) = b.(bf{i});
end
