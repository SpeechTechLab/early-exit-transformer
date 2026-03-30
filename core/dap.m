function [an,e] = dap(x,p,varargin)
% DAP Discrete All-Pole Model.
% A = DAP(X,P) finds the coefficients, A=[ 1 A(2) ... A(P+1) ],
% of an Nth order discrete all-pole model filter
%
%    Xp(n) = -A(2)*X(n-1) - A(3)*X(n-2) - ... - A(P+1)*X(n-P)
%
% such that a discrete version of Itakura-Saito distortion measure
% is minimized.
%
% X can be a vector or a matrix. If X is a matrix containing a
% separate signal in each column, DAP returns a model estimate for
% each column in the rows of A. P specifies the order of the
% polynomial A(z).
%
% If you do not specify a value for P, DAP uses a default P = length(X)-1.
%
% [A,E] = DAP(X,P) returns the variance (power) of the prediction error.
% [A,E,F] = DAP(X,P) returns the list of harmonic peaks used in
% optimisation.
%
% If X is non-periodic, DAP returns A=1 and E=-1.
%
% DAP uses the method proposed by El-Jaroudi and Makhoul (1991)
% to solve the coefficients.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~isa(x,'signal')
  x = signal(x,1);
end
fs = x.time.fs;

% variable initializations

maxiter = 300;
iter = 0;
% iteration threshold
T = 0.0001;
% max change per iterations
alpha = 0.5;
% DAP seed
seed = 'enh';
f0 = -1;
% autocorrelation estimation type
autoctype = 'ft'; % default: Fourier Transform
%autoc = 'mv'; % testing minimum variance

if nargin>2
  options = varargin{1};
  if isfield(options,'maxiter')
    maxiter = options.maxiter;
  end
  if isfield(options,'iter')
    iter = options.iter;
  end
  if isfield(options,'thres')
    T = options.thres;
  end
  if isfield(options,'alpha')
    alpha = options.alpha;
  end
  if isfield(options,'seed')
    seed = options.seed;
  end
  if isfield(options,'aropt')
    for i = fieldnames(options.aropt)
      aropt.(i) = options.aropt.i;
    end
  end
  if isfield(options,'f0')
      f0 = options.f0;
  end
  if isfield(options,'flist')
      f = options.flist;
  end
  if isfield(options,'xcorr')
      autoctype = options.xcorr;
  end
end

% get discrete frequencies
if ~exist('f','var') || isempty(f)
    f = find_f0(signal(x, fs))';
end

% check frequency list; is equal to null?
if isempty(f)
    % non-existent frequency list -> abort 
    an = [1 zeros(1,p)];
    e = -1;
    return
end

% create Fourier matrix C
C = exp(-j*(2*pi*f/fs)*(0:p));

% compute energy
if strcmp(autoctype,'ft') % fourier transform
    [H, f_temp] = freqz(x.s,1,512,fs);
    P = abs(H).^2;
elseif strcmp(autoctype,'mv') % minimum variance estimate
    R = xcorr(x,p);
    Rmv = toeplitz(R(1+p+(0:p)));
    iRmv = inv(Rmv);
    for k=1:length(f)
        P(k,1) = real(1./(C(k,:)*iRmv*C(k,:)'));
    end
end

N=length(f); % number of discrete frequencies

% calculate the autocorrelation coefficients
% El-Jaroudi 91 Eq. (5)
r = zeros(p,1);
for i=0:p
  r(i+1) = real(1/N * sum(P.*exp(j*(2*pi*f/fs)*i)));
end

eIS = [];

Rm = real(toeplitz(r));

% while singular
while min(abs(eig(Rm))) < max(size(Rm))*norm(Rm)*eps
  % add white noise until non-singular
  Rm = Rm + Rm(1)*0.01*eye(size(Rm));
end

invRm = inv(Rm); % FIXME: replace this with levinson-durbin

if any(isinf(invRm(:)))
  warning('Invalid invRm, setting to zero.');
  invRm=zeros(size(invRm));
end


if strcmp(seed,'lpc')
  % seed the loop with regular LPC
  a = real(lpc(x.s,p));
elseif strcmp(seed,'enh') 
  a = [1 zeros(1,p)]*invRm;
else
  error('Invalid value for seed parameter');
end

% create composite matrix D
D = (1./N)*invRm*C.';


% start the loop
while itercrit(eIS,maxiter,iter,T)
    
    % Denominator of Eq. (36) step 4
    Awm = C*a'; 
    
    a = a*(1-alpha) +  alpha*real(D*(1./Awm))'; 
  
    % calculate Phat for Itakura-Saito error measure
  
%    [Hhat,what] = freqz(1,a,f,'whole',fs);
%    Phat = abs(Hhat).^2;
    Phat = abs(1./(C*a')).^2;
  
    % now calculate Itakura-Saito error measure (El-Jaroudi Eq. (14))

    eIS(length(eIS)+1) = 1/N * sum(P./Phat-log(P./Phat)-1);
end
e=eIS(end);
%disp(['Number of iterations: ' num2str(length(e)) ]);

% normalize the coefficients

an=a/a(1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [y]=itercrit(eIS,maxiter,iter,T)

y=0;
if iter~=0
  if length(eIS)<iter
    y=1;
  end
else
  if (length(eIS)<2 || abs(eIS(length(eIS)-1)-eIS(length(eIS))) > T) ...
	& length(eIS)<maxiter
    y=1;
  end
end

