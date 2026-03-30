function metrics = compute_glottal_metrics(g, fs, f0)
% COMPUTE_GLOTTAL_METRICS Computes NAQ, QOQ, HRF, and H1-H2 for a glottal flow signal.
%
%   metrics = compute_glottal_metrics(g, fs, f0)
%
%   Inputs:
%       g  : Glottal flow signal (time domain vector)
%       fs : Sampling frequency (Hz)
%       f0 : Fundamental frequency (Hz)
%
%   Outputs:
%       metrics : Struct containing:
%           .NAQ : Normalized Amplitude Quotient
%           .QOQ : Quasi-Open Quotient
%           .HRF : Harmonic Richness Factor (dB)
%           .H1H2: H1-H2 (dB)

    % Handle signal objects or structs
    if isa(g, 'signal') || (isstruct(g) && isfield(g, 's'))
        g = g.s;
    end
    g = g(:);
    
    % 1. Basic Parameters
    T0_sec = 1/f0;
    T0_samples = round(fs/f0);
    
    % Ensure we have enough data (at least one period)
    if length(g) < T0_samples
        metrics.NAQ = NaN;
        metrics.QOQ = NaN;
        metrics.HRF = NaN;
        metrics.H1H2 = NaN;
        return;
    end

    % 2. Time Domain Metrics (NAQ, QOQ)
    
    % Differentiate flow to get derivative (differentiated flow)
    % Scale by fs to get physical units consistency (though ratios cancel out)
    dg = [diff(g); 0] * fs; 
    
    % Detrend g for robust amplitude calculation
    g_detrend = detrend(g);
    
    % A_ac: Peak-to-peak amplitude of flow
    A_ac = max(g_detrend) - min(g_detrend);
    
    % E_e: Maximum excitation (negative peak of derivative)
    % We look for the minimum value of the derivative (most negative).
    % To be robust against noise, we can use findpeaks on -dg.
    min_peak_height = max(abs(dg)) * 0.2;
    
    % Temporarily disable the warning
    warning('off', 'signal:findpeaks:largeMinPeakHeight');
    
    if isempty(dg) || max(-dg) <= min_peak_height
        pks = [];
    else
        [pks, ~] = findpeaks(-dg, 'MinPeakHeight', min_peak_height, 'MinPeakDistance', round(T0_samples*0.8));
    end
    
    % Re-enable the warning
    warning('on', 'signal:findpeaks:largeMinPeakHeight');
    
    if isempty(pks)
        E_e = max(abs(dg)); % Fallback to absolute max if no clear peaks
    else
        E_e = mean(pks); % Average of detected excitation peaks
    end
    
    % NAQ Calculation
    if E_e > 0
        metrics.NAQ = A_ac / (E_e * T0_sec);
    else
        metrics.NAQ = NaN;
    end
    
    % QOQ Calculation
    % Duration where flow > 50% of amplitude
    threshold = min(g_detrend) + 0.5 * A_ac;
    above_thresh = g_detrend > threshold;
    
    % Find lengths of consecutive segments above threshold
    d_above = diff([0; above_thresh; 0]);
    starts = find(d_above == 1);
    ends = find(d_above == -1);
    
    if ~isempty(starts) && ~isempty(ends)
        lengths = ends - starts;
        % Filter lengths to be within reasonable bounds (10% to 100% of period)
        valid_mask = (lengths > 0.1*T0_samples) & (lengths < 1.1*T0_samples);
        if any(valid_mask)
            avg_len = mean(lengths(valid_mask));
            metrics.QOQ = avg_len / T0_samples;
        else
            metrics.QOQ = NaN;
        end
    else
        metrics.QOQ = NaN;
    end
    
    % 3. Frequency Domain Metrics (HRF, H1-H2)
    % Apply Hamming window
    w = hamming(length(g));
    G = fft(g .* w, 2^nextpow2(4*length(g)));
    mag = abs(G);
    freqs = (0:length(mag)-1) * fs / length(mag);
    
    % Find H1
    [h1_val, ~] = find_peak_near(mag, freqs, f0, 0.1*f0);
    
    % Find H2
    [h2_val, ~] = find_peak_near(mag, freqs, 2*f0, 0.1*f0);
    
    % HRF: Sum of harmonics / H1
    harmonics_sum = 0;
    k = 2;
    while k*f0 < fs/2
        [hk_val, ~] = find_peak_near(mag, freqs, k*f0, 0.1*f0);
        harmonics_sum = harmonics_sum + hk_val;
        k = k + 1;
    end
    
    if h1_val > 0
        metrics.HRF = 20 * log10(harmonics_sum / h1_val);
        metrics.H1H2 = 20 * log10(h1_val) - 20 * log10(h2_val);
    else
        metrics.HRF = NaN;
        metrics.H1H2 = NaN;
    end

end

function [val, idx] = find_peak_near(spectrum, freqs, target_freq, bandwidth)
    mask = (freqs >= target_freq - bandwidth) & (freqs <= target_freq + bandwidth);
    if ~any(mask)
        [~, closest_idx] = min(abs(freqs - target_freq));
        val = spectrum(closest_idx);
        idx = closest_idx;
    else
        range_indices = find(mask);
        [val, local_idx] = max(spectrum(range_indices));
        idx = range_indices(local_idx);
    end
end
