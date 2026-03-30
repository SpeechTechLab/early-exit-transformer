function y = win(x,winfhandle,varargin)
% WIN  Window a signal using a given window.
%   WIN(X,FHND,VARARGIN) returns signal X windowed using the window
%   function given in handle FHND. Excess arguments are given to
%   the function handle.

% $Id: win.m 3 2004-02-04 12:57:04Z mairas $

if isa(x, 'signal')
    x_data = x.s;
    fs = x.fs;
    w = feval(winfhandle,length(x_data),varargin{:});
    y_data = w(:)' .* x_data(:)';  % Ensure both are row vectors
    y = signal(y_data, fs);
else
    w = feval(winfhandle,length(x),varargin{:});
    y = w(:)' .* x(:)';  % Ensure both are row vectors
end
