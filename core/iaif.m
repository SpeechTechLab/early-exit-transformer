function [g,Hvt2,e_ar,Hg2]=iaif(s,varargin)
% IAIF  Inverse filtering of a signal using DAP or LPC.
%   [G,HVT2,E,HG2,IAIFOPT]=IAIF(S,FS,VARARGIN) where S is a vector
%   [G,HVT2,E,HG2,IAIFOPT]=IAIF(S,VARARGIN) where S is a SIG object.
%   G is a signal object with the glottal flow, HVT2 the corresponding
%   AR-model, E the Itakura-Saito error for the final model, HG2 the
%   ar-model of the intermediate glottal flow approximation and IAIFOPT a
%   structure with the IAIF model parameters.

% $Id: iaif.m 200 2007-09-10 12:32:40Z mairas $

argin = varargin;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check parameters

retsig=1;
if ~isa(s,'signal')
  fs = argin{1} ; argin = argin(2:end);
  s=signal(s,fs);
  retsig=0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set the parameter defaults

% order of the LPC/DAP
p = get_if_order(s.fs);
r = p;
g = 4;
automodel = false;

% AR default options

aropt = struct;
options = struct;

% integrator coeff
rho = 0.99;
causality = 'noncausal';

% window function
winfunc = @hamming;

% autoregressive function used
arfunc = 'dap';

% return integrated output
diffout = 0;

% remove positive real poles
remove_real_poles = 0;

% f0 is needed to accurately estimate the delays caused by the filter
% operations
f0 = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read the options

if ~isempty(argin)
  options = argin{1};
  if isfield(options,'p')
    p = options.p;
    r = p;
    automodel = false;
  end
  if isfield(options,'automodel')
      automodel = options.automodel;
  end
  if isfield(options,'r')
    r = options.r;
  end
  if isfield(options,'g')
    g = options.g;
  end
  if isfield(options,'rho')
    rho = options.rho;
  end
  if isfield(options,'winfunc')
    winfunc = options.winfunc;
  end
  if isfield(options,'arfunc')
    arfunc = options.arfunc;
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

% if p~=r
%   warning(['iaif: p=' num2str(p) ' differs from r=' num2str(r)]);
% end

% if this is not self-recursion, then clear frequency list
if automodel ~= -1
    aropt.flist = [];
end

% make separate options for glottal model and vocal tract model
aropt_g = aropt;
aropt_vc = aropt;
arfunc_vc = arfunc;
if isfield(options,'g_model')
    aropt_g.xcorr = options.g_model;
else
    aropt_g.xcorr = 'ft';
end
if isfield(options,'vc_model')
    aropt_vc.xcorr = options.vc_model;
else
    aropt_vc.xcorr = 'ft';
end
if isfield(options,'arfunc_g')
    arfunc_g = options.arfunc_g;
else
    arfunc_g = 'lpc';
end
if isfield(options,'arfunc_vc')
    arfunc_vc = options.arfunc_vc;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% automatic model order specification
if automodel == 1
    % check if any input options were specified
    if exist('options','var')
        s_options = options;  % options for iaif in iteration
    end
    if ~exist('s_options','var');
        s_options = struct;
    end
    s_options.automodel = -1; % flag for self-recursion
    
    [g,Hvt2,e_ar,Hg2,options] = auto_iaif(s,s_options,p);

    return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% processing begins here

% note that the highpass filtering (block 1) needs to be done in advance!
sw = win(s,winfunc);

% perform a 1st-order DAP analysis on the filter to create a
% preliminary estimate for the effect of the glottal pulseform on
% the speech signal (block 2)
[Hg1, e_1]=ar(arfunc_g,sw,1,aropt_g);

% inverse filter (block 3)
% Always use causal inverse filtering for physical modeling
sg1_temp = filter(Hg1,1,s.s);
sg1 = valid(signal(sg1_temp, s.fs));
sg1w=win(sg1,winfunc);

% calculate order p DAP analysis (block 4)
[Hvt1, e_2] = ar(arfunc_vc,sg1w,p,aropt_vc);

% remove positive real poles
%[z,p,k] = tf2zp(1,Hvt1);
%p_ = p(p<0 | abs(imag(p))>1e-15);
%[dummy,Hvt1] = zp2tf(z,p_,k);


% inverse filter (block 5)
% Always use causal inverse filtering
g1_temp = filter(Hvt1,1,s.s);
g1 = valid(signal(g1_temp, s.fs));

% cancel the lip-radiation effect by integrating svt1 (block 6)
u1=integrate(g1,rho,f0,causality);
u1w=win(u1,winfunc);

% calculate order g DAP analysis to get a better estimate for the
% contribution of the glottal flow (block 7)
[Hg2, e_dap] = ar(arfunc_g,u1w,g,aropt_g);

% inverse filter s (block 8)
% Always use causal inverse filtering
sg2_temp = filter(Hg2,1,s.s);
sg2 = valid(signal(sg2_temp, s.fs));

% integrate (block 9)
ug2=integrate(sg2,rho,f0,causality);
ug2w=win(ug2,winfunc);


% calculate order r DAP analysis to get a new model of the vocal
% tract (block 10)
[Hvt2c, e_ar] = ar(arfunc_vc,ug2w,r,aropt_vc);
Hvt2 = real(Hvt2c);

% normalize
Hvt2 = Hvt2/sum(Hvt2);

% remove positive real poles
if remove_real_poles
  [z,p,k] = tf2zp(1,Hvt2);
  p_ = p(p<0 | abs(imag(p))>1e-15);
  [dummy,Hvt2] = zp2tf(z,p_,k);
end

% inverse filter (block 11)
% Always use causal inverse filtering
dg_temp = filter(Hvt2,1,s.s);
dg = valid(signal(dg_temp, s.fs));

if diffout==1
  g = dg;
else
  % integrate (block 12)
  g_us = integrate(dg,rho,f0,causality);

  % normalize
  if isa(g_us, 'signal')
    g_data = g_us.s;
    g = signal(g_data - min(g_data), g_us.fs);
  else
    g = g_us - min(g_us);
  end
end

if ~retsig
  g = g.s;
end

% return options
options = mergestruct(aropt,options);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [y,e_ar]=ar(arfunc,x,p,opt)

if isa(x, 'signal')
    x = x.s;
end

if strcmp(arfunc,'dap')
  [y,e_ar] = dap(x,p,opt);
elseif strcmp(arfunc,'lpc')
  [y,e_ar] = lpc(x,p);
  y = real(y);
  f_ar = [];
elseif strcmp(arfunc,'mvdr')
  [y,e_ar] = mvdr(x,p);
  f_ar = [];    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y=integrate(x,rho,f0,causality)
% Extract data if x is a signal object
if isa(x, 'signal')
    x_data = x.s;
    fs = x.fs;
    is_signal = true;
else
    x_data = x;
    is_signal = false;
end

if strcmp(causality, 'causal')
   y_data = filter(1,[1 -rho],x_data);
else
   y_data = -flip(filter(1,[1 -rho],flip(x_data)));
end

% Return signal object if input was signal object
if is_signal
    y = signal(y_data, fs);
else
    y = y_data;
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

