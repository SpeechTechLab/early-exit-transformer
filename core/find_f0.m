function f0 = find_f0(x, fs)
% FIND_F0  Find fundamental frequency
%
%    F0 = FIND_F0(X)
%
%    Find the fundamental frequency of signal X. 
%    By default, FIND_F0 uses the YIN method for F0 estimation.

% $Id: find_f0.m 127 2007-11-07 14:22:23Z mairas $


f0 = find_f0_yin(x);
if (f0 == 0) || (x.fs == 0) || isnan(f0)
    % no F0 found
    return;
end

% F0 Sanity check
% - periodicity is at least 0.5 energy of total energy

% autocorrelation length
len = ceil(x.fs/f0);

% window (number of samples) around F0 peak in autocorrelation
autocwindow = round(len/8);

% calculate autocorrelation
R=xcorr(x.s,len+autocwindow,'coeff');
R = R(1+len+autocwindow+(0:(len+autocwindow)));

% check if peak near F0 is at least 0.3 of signal energy
if R(1) > max(R(1+len+(-autocwindow:autocwindow)))/0.3
    % check integer multiples 2, 3 and 4
    if (autocwindow*2 < len/4) && ...
            (R(1) > max(R(1+round(len/4)+(-autocwindow:autocwindow)))/0.5)
        f0 = f0*4;
    elseif (autocwindow*2 < len/3) && ...
            (R(1) > max(R(1+round(len/3)+(-autocwindow:autocwindow)))/0.5) 
        f0 = f0*3;
    elseif (autocwindow*2 < len/2) && ...
            (R(1) > max(R(1+round(len/2)+(-autocwindow:autocwindow)))/0.5) 
        f0 = f0*2;
    else
        disp(['F0 Sanity failed, with F0: ' num2str(f0) 'Hz']);
        % too small - replace F0 by zero
        f0 = 0;
    end
end