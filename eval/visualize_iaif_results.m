function visualize_iaif_results(x, fs, results, frame_indices, analysis_mode)
    % Visualize frame-by-frame IAIF analysis results
    
    figure('Position', [50, 50, 1600, 1000]);
    
    % 1. Original signal with frame markers
    subplot(5, 2, [1, 2]);
    t = (0:length(x)-1) / fs;
    plot(t, x, 'Color', [0.5, 0.5, 0.5]);
    hold on;
    
    % Mark frame positions
    valid_results = cellfun(@(r) ~isempty(r) && r.success, results);
    valid_indices = frame_indices(valid_results);
    y_lim = [min(x), max(x)];
    
    for i = 1:min(20, length(valid_indices))
        x_pos = valid_indices(i)/fs;
        plot([x_pos, x_pos], y_lim, 'r--', 'LineWidth', 0.5);
    end
    
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(sprintf('Original Signal with Frame Markers (%s mode)', analysis_mode));
    grid on;
    hold off;
    
    % 2. Example frame - original
    subplot(5, 2, 3);
    frame_to_show = round(length(results) / 2);
    if ~isempty(results{frame_to_show}) && results{frame_to_show}.success
        frame_start = frame_indices(frame_to_show);
        frame_end = frame_start + results{frame_to_show}.frame_length - 1;
        t_frame = (frame_start:frame_end) / fs;
        plot(t_frame, x(frame_start:frame_end));
        xlabel('Time (s)');
        ylabel('Amplitude');
        title(sprintf('Frame %d - Original', frame_to_show));
        grid on;
    end
    
    % 3. Example frame - glottal flow
    subplot(5, 2, 4);
    if ~isempty(results{frame_to_show}) && results{frame_to_show}.success
        t_g = (0:length(results{frame_to_show}.g)-1) / fs;
        plot(t_g, results{frame_to_show}.g);
        xlabel('Time (s)');
        ylabel('Amplitude');
        title(sprintf('Frame %d - Glottal Flow', frame_to_show));
        grid on;
    end
    
    % 4. Reconstructed glottal flow (OLA) - reuses reconstruct_signal_generic
    subplot(5, 2, [5, 6]);
    glottal_concat = reconstruct_signal(results, frame_indices, length(x), 'g', fs);
    t_concat = (0:length(glottal_concat)-1) / fs;
    plot(t_concat, glottal_concat);
    xlabel('Time (s)');
    ylabel('Amplitude');
    title('Reconstructed Glottal Flow (All Frames)');
    grid on;
    
    % 5. Vocal tract filters across frames (reuses plot_filter_evolution)
    subplot(5, 2, 7);
    plot_filter_evolution(results, fs, 'Hvt', 'Vocal Tract');
    
    % 6. Glottal filters across frames
    subplot(5, 2, 8);
    plot_filter_evolution(results, fs, 'Hg', 'Glottal');
    
    % 7. AR residual example
    subplot(5, 2, 9);
    if ~isempty(results{frame_to_show}) && results{frame_to_show}.success
        t_ar = (0:length(results{frame_to_show}.e_ar)-1) / fs;
        plot(t_ar, results{frame_to_show}.e_ar);
        xlabel('Time (s)');
        ylabel('Amplitude');
        title(sprintf('Frame %d - AR Residual', frame_to_show));
        grid on;
    end
    
    % 8. Reconstructed AR residual
    subplot(5, 2, 10);
    ar_concat = reconstruct_signal(results, frame_indices, length(x), 'e_ar', fs);
    t_ar_concat = (0:length(ar_concat)-1) / fs;
    plot(t_ar_concat, ar_concat);
    xlabel('Time (s)');
    ylabel('Amplitude');
    title('Reconstructed AR Residual (All Frames)');
    grid on;
    
    set(gcf, 'Color', 'w');
end