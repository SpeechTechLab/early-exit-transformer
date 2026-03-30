function [a,g] = lpc_signal(x,p)
% Wrapper for lpc that handles signal objects

if isa(x, 'signal')
    [a,g] = lpc(x.s,p);
else
    [a,g] = lpc(x,p);
end