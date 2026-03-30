function pitch_info = estimate_pitch(x, fs)
    % Simple pitch estimation using autocorrelation
    % You can replace with more sophisticated methods (YIN, SWIPE, etc.)
    
    frame_length = round(0.04 * fs);  % 40ms window
    frame_shift = round(0.01 * fs);   % 10ms shift
    
    min_lag = round(fs / 500);  % Max 500 Hz
    max_lag = round(fs / 50);   % Min 50 Hz
    
    num_frames = floor((length(x) - frame_length) / frame_shift) + 1;
    f0 = zeros(num_frames, 1);
    
    for i = 1:num_frames
        start_idx = (i-1) * frame_shift + 1;
        end_idx = start_idx + frame_length - 1;
        
        if end_idx > length(x)
            break;
        end
        
        frame = x(start_idx:end_idx);
        
        % Apply window
        frame = frame .* hamming(length(frame));
        
        % Autocorrelation
        [acf, lags] = xcorr(frame, max_lag, 'coeff');
        acf = acf(lags >= min_lag);
        lags_pos = lags(lags >= min_lag);
        
        % Find peak
        [~, peak_idx] = max(acf);
        period = lags_pos(peak_idx);
        
        if period > 0
            f0(i) = fs / period;
        end
    end
    
    % Median filtering for smoothing
    f0 = medfilt1(f0, 5);
    
    pitch_info.f0 = f0;
    pitch_info.periods = fs ./ f0;
    pitch_info.frame_shift = frame_shift;
end