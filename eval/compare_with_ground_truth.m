function compare_with_ground_truth(x, fs, results, frame_indices, method_name, gt_filepath)
    % 1. Load Ground Truth
    [gt, fs_gt] = audioread(gt_filepath);
    if size(gt, 2) > 1, gt = mean(gt, 2); end
    gt = gt(:);
    if fs_gt ~= fs, gt = resample(gt, fs, fs_gt); end
    
    % 2. Reconstruct Estimated Signal
    if strcmp(method_name, 'QCP')
        field_name = 'sg';
    elseif strcmp(method_name, 'IAIF')
        field_name = 'g';
    elseif strcmp(method_name, 'TRLP')
        field_name = 'g';
    else
        error('Unknown method');
    end
    
    estimated = reconstruct_signal(results, frame_indices, length(x), field_name, fs);
    
    % Post-process Estimated: Detrend and mild Low-pass
    estimated = detrend(estimated, 'linear'); % Remove DC drift
    [b,a] = butter(2, 2000/(fs/2), 'low');    % Clean up high freq noise for comparison
    estimated = filtfilt(b,a, estimated);

    % 3. Align Lengths
    L = min(length(gt), length(estimated));
    gt = gt(1:L);
    estimated = estimated(1:L);
    
    % Remove DC from GT for fair comparison with AC-coupled estimate
    gt = gt - mean(gt);

    % 4. TIME ALIGNMENT (Crucial step)
    % Find best lag
    max_lag = round(0.02 * fs); % Allow +/- 20ms shift
    [r, lags] = xcorr(gt, estimated, max_lag);
    [~, I] = max(abs(r));
    lag_diff = lags(I);
    
    % Shift estimated signal
    if lag_diff > 0
        estimated = [zeros(lag_diff, 1); estimated(1:end-lag_diff)];
    elseif lag_diff < 0
        estimated = [estimated(-lag_diff+1:end); zeros(-lag_diff, 1)];
    end
    
    % 5. Check Inversion
    % If correlation is negative, flip the signal
    tmp_corr = corr(gt, estimated);
    if tmp_corr < 0
        estimated = -estimated;
        fprintf('Auto-flipping inverted signal for %s.\n', method_name);
    end
    
    % 6. Normalize to [0, 1]
    % Robust Normalization: Ignore first and last 5% for amplitude calculation to avoid edge transients
    trim_percent = 0.05;
    trim_s = max(1, round(L * trim_percent));
    trim_e = min(L, round(L * (1 - trim_percent)));
    valid_range = trim_s:trim_e;
    
    if isempty(valid_range), valid_range = 1:L; end
    
    % Normalize GT to [0, 1]
    gt_min = min(gt(valid_range));
    gt_max = max(gt(valid_range));
    gt_norm = (gt - gt_min) / (gt_max - gt_min);
    
    % Normalize Estimated to [0, 1]
    est_min = min(estimated(valid_range));
    est_max = max(estimated(valid_range));
    estimated_norm = (estimated - est_min) / (est_max - est_min);
    
    % 7. Metrics
    correlation = corr(gt_norm, estimated_norm);
    rmse = sqrt(mean((gt_norm - estimated_norm).^2));
    
    fprintf('\n=== %s Metrics (Aligned) ===\n', method_name);
    fprintf('Correlation: %.4f\n', correlation);
    fprintf('RMSE:       %.4f\n', rmse);
    
    % === PLOTTING ===
    figure('Position', [100, 100, 1400, 600]);
    t = (0:L-1)/fs;
    
    subplot(3, 1, 1);
    plot(t, gt_norm, 'b'); hold on;
    plot(t, estimated_norm, 'r-'); % Solid red line
    title([method_name ' Aligned Comparison']); legend('GT', 'Est'); grid on;
    
    subplot(3, 1, 2);
    % Zoom in middle
    mid = round(L/2); 
    idx = mid:min(mid+400, L);
    plot(t(idx), gt_norm(idx), 'b', 'LineWidth', 1.5); hold on;
    plot(t(idx), estimated_norm(idx), 'r-', 'LineWidth', 1.5); % Solid red line
    title('Zoomed View'); grid on;
    
    subplot(3,1,3);
    pwelch(estimated_norm, [], [], [], fs);
    title('PSD of Estimate');
end