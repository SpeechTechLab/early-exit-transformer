function h1h2 = compute_h1h2(signal, fs, f0)
    % Compute H1-H2 (in dB) for a given signal frame.
    % H1: Magnitude of the first harmonic (fundamental frequency)
    % H2: Magnitude of the second harmonic (2 * f0)
    
    % Ensure column vector
    signal = signal(:);
    
    % Apply Hamming window
    N = length(signal);
    w = hamming(N);
    signal_w = signal .* w;
    
    % Compute FFT
    Nfft = 2^nextpow2(N * 4); % Zero padding for better resolution
    X = fft(signal_w, Nfft);
    f = (0:Nfft-1) * fs / Nfft;
    
    % Magnitude in dB
    X_mag = abs(X);
    X_db = 20 * log10(X_mag + eps);
    
    % Find peak near f0 (H1)
    search_bw = 0.1 * f0; % Search bandwidth +/- 10%
    [h1_val, h1_idx] = find_peak_near(X_db, f, f0, search_bw);
    
    % Find peak near 2*f0 (H2)
    [h2_val, h2_idx] = find_peak_near(X_db, f, 2*f0, search_bw);
    
    h1h2 = h1_val - h2_val;
end

function [val, idx] = find_peak_near(spectrum, freqs, target_freq, bandwidth)
    % Find the maximum value in the spectrum within target_freq +/- bandwidth
    
    mask = (freqs >= target_freq - bandwidth) & (freqs <= target_freq + bandwidth);
    if ~any(mask)
        % Fallback: find closest frequency bin
        [~, closest_idx] = min(abs(freqs - target_freq));
        val = spectrum(closest_idx);
        idx = closest_idx;
    else
        % Find max in the range
        range_indices = find(mask);
        [val, local_idx] = max(spectrum(range_indices));
        idx = range_indices(local_idx);
    end
end
