function [sg,Hvt,e_ar,Hg]=qcp(s,varargin)
%QCP Summary of this function goes here
%   Detailed explanation goes here

argin = varargin;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check parameters

retsig=1;
if ~isa(s,'signal')
  fs = argin{1} ; argin = argin(2:end);
  s=signal(s,fs);
  retsig=0;
end
fs = s.fs;
p = get_if_order(s.fs);
g = 4;

% AR default options

aropt = struct;
options = struct;

% integrator coeff
rho = 0.99;
causality = 'noncausal';

% DQ and PQ
%DQ = 0.7;
%PQ = 0.1;

% window function
winfunc = @hamming;

% remove positive real poles
remove_real_poles = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read the options

if ~isempty(argin)
  options = argin{1};
  if isfield(options,'p')
    p = options.p;
    automodel = false;
  end
  if isfield(options,'automodel')
      automodel = options.automodel;
  end
  if isfield(options,'g')
    g = options.g;
  end
  if isfield(options,'rho')
    rho = options.rho;
  end
  if isfield(options,'dq')
    DQ = options.dq;
  end
  if isfield(options,'pq')
    PQ = options.pq;
  end
  if isfield(options,'nramp')
    Nramp = options.nramp;
  end
  if isfield(options,'winfunc')
    winfunc = options.winfunc;
  end
  if isfield(options,'aropt')
    aropt = mergestruct(aropt,options.aropt);
  end
  if isfield(options,'diffout')
    diffout = options.diffout;
  end
  if isfield(options,'f0')
    aropt.f0 = options.f0;
  else
    aropt.f0 = find_f0(s);
  end
  if isfield(options,'flist') 
    aropt.flist = options.flist;
  end
  if isfield(options,'remove_real_poles') 
    remove_real_poles = options.remove_real_poles;
  end
  if isfield(options,'causality')
      if options.causality == 1
          causality = 'causal';
      else
          causality = 'noncausal';
      end
  end
         
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% processing begins here
x = s.s;
f0 = aropt.f0;
% note that the highpass filtering (block 1) needs to be done in advance!
[gci_ins goi_ins] = gci(x,aropt.f0,fs); % GCI estimation
wmin = 0.00001; % Minimum value of weighting function
%DQ = 0.7;  Duration Quotient (relative to fundamental period) 0.4 -- 1
%PQ = 0.1; % Position Quotient (relative to fundamental period) 0.0, 0.05, 0.1
%Nramp = round(fs/8000*7); % Length of linear ramp (in samples)

w = makeW(x,p,DQ,PQ,wmin,Nramp,gci_ins,fs); % Use this to manually select PQ and DQ values
%w = makeW_adapt(x,p,gci_ins,f0); % Use this for predefined, pitch-adaptive values for weighting function

s2 = filter([1 -1],1,x); % Pre-emphasis
sw = win(s2,winfunc);
[Hvt, e_ar] = wlp(sw,w,p);
Hvt = Hvt(:)';

if remove_real_poles
  [z,p,k] = tf2zp(1,Hvt);
  p_ = p(p<0 | abs(imag(p))>1e-15);
  [dummy,Hvt] = zp2tf(z,p_,k);
end

% Inverse filtering should always be causal to cancel the causal vocal tract
% We filter the original signal 'x' (not pre-emphasized 's2') to get the glottal flow derivative.
dg_temp = filter(Hvt,1,x);

% Adjust sign for integration
if strcmp(causality, 'causal')
    % Forward integration requires input U' to yield U
    % dg_temp is prediction error (~ -U' if s2 is -S')
    % So we negate it to get U'
    dg_temp = -dg_temp;
else
    % Backward integration requires input -U' to yield U
    % dg_temp is ~ -U'
    % So we keep it as is
    dg_temp = dg_temp;
end

dg = valid(signal(dg_temp, s.fs));
sg=integrate(dg,rho,f0,causality);
Hg = lpc_signal(win(sg,winfunc),g);


end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y=integrate(x,rho,f0,causality)
% Extract data if x is a signal object
    if isa(x, 'signal')
        x_data   = x.s(:);
        fs       = x.fs;
        is_signal = true;
    else
        x_data   = x(:);
        fs       = [];
        is_signal = false;
    end

    if strcmpi(causality, 'causal')
        % One?sided leaky integrator
        y_data = filter(1, [1 -rho], x_data);
    else
        % Symmetric (non?causal) integrator � NO extra sign flip
        y_data = flip(filter(1, [1 -rho], flip(x_data)));
    end

    if is_signal
        y = signal(y_data, fs);
    else
        y = y_data;
    end
end
%y=filter(1,[1 -rho],x,'f0',f0,causality);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p = get_if_order(fs)

% GET_IF_ORDER  Get suitable inverse filter order for the sampling frequency
%
%  P = GET_IF_ORDER(FS)
%  Returns a suitable order for DAP/LPC inverse filter for a given
%  FS.

p = round(fs/1000)+2;
end

function [ w ] = makeW( x, p, DQ, PQ, d, Nramp, gci_ins, fs )
%   Create a AME weight function for frame x for LPC order p
%   DQ = duration quotient (from 0 to 1)
%   PQ = Position Quotient (from 0 to 1)
%   d = minimum value of the weight function
%   Nramp = length of the linear ramp (in samples)
%   gci_ins = Glottal Closure Instants of the frame x


N = length(x);
%Nramp = 1;
if Nramp > 0
UPramp = linspace(d,1,2+Nramp);
UPramp = UPramp(2:end-1);
DOWNramp = UPramp(end:-1:1);
end


if DQ+PQ > 1
    DQ = 1-PQ;
end

w = d.*ones(1,N+p);



for i = 1:length(gci_ins)-1
   T = gci_ins(i+1)-gci_ins(i);
   T1 = round(DQ*T);
   T2 = round(PQ*T);
   while T1+T2 > T
       T1 = T1-1;
   end
   w(gci_ins(i)+T2:gci_ins(i)+T2+T1-1) = 1;
   if Nramp > 0
       w(gci_ins(i)+T2:gci_ins(i)+T2+Nramp-1) = UPramp;
       if gci_ins(i)+T2+T1-Nramp > 0
           w(gci_ins(i)+T2+T1-Nramp:gci_ins(i)+T2+T1-1) = DOWNramp;
       end
   end
end

Nend = N-(T2+gci_ins(i+1));

if T2+gci_ins(i+1) < N
    if T1+T2 < Nend
        w(gci_ins(i+1)+T2:gci_ins(i+1)+T2+T1-1) = 1;
        if Nramp > 0
            w(gci_ins(i+1)+T2:gci_ins(i+1)+T2+Nramp-1) = UPramp;
            w(gci_ins(i+1)+T2+T1-Nramp:gci_ins(i+1)+T2+T1-1) = DOWNramp;
        end
    else
        T1 = Nend-T2;
                w(gci_ins(i+1)+T2:gci_ins(i+1)+T2+T1-1) = 1;
        if Nramp > 0
            w(gci_ins(i+1)+T2:gci_ins(i+1)+T2+Nramp-1) = UPramp;
        end
    end
end


end

